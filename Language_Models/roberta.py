#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoBERTa on SQuAD v1.1 with RepQ PTQ + ViT-style parallel adapters
+ INTx fake-quantized embeddings (word/pos/segment).

- Model: AutoModelForQuestionAnswering (RoBERTa)
- Data: SQuAD v1.1
- Quant: RepQ (quant_model + set_quant_state) + per-row INTx embeddings
- Adapters: DepthwiseSeparableAdapter1D (attn) + MLPTokenAdapter (mlp), parallel add (pre-LN)
- Train: freeze backbone; optimize ONLY adapters (KD + CE on start/end)
- Metrics: EM/F1 on SQuAD dev
"""

import os, sys, math, random, argparse, inspect
from typing import Optional, Tuple, List, Literal, Dict, Any

# ---- Make parent dir importable (for your RepQ + utils) ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quant.quant_model import quant_model, set_quant_state  # RepQ hooks
from utils.utils import compute_quantized_params  # optional size counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    default_data_collator,
)

# ----------------------- Repro -----------------------
def seed_everything(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ---------------- Quantized Embeddings (per-row affine) ----------------
class QuantEmbedding(nn.Embedding):
    """
    Fake-quantize embedding WEIGHTS (per-row affine, INTx). Stores FP weights and
    applies quantize-dequant on forward. Calibrates once via min/max per row.
    """
    def __init__(self, num_embeddings, embedding_dim, *, n_bits=4, per_row=True, **kw):
        super().__init__(num_embeddings, embedding_dim, **kw)
        self.n_bits = int(n_bits); self.per_row = bool(per_row)
        if per_row:
            self.register_buffer("scale", torch.ones(num_embeddings), persistent=True)
            self.register_buffer("zp",    torch.zeros(num_embeddings), persistent=True)
        else:
            self.register_buffer("scale", torch.ones(1), persistent=True)
            self.register_buffer("zp",    torch.zeros(1), persistent=True)
        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.uint8), persistent=True)

    @torch.no_grad()
    def calibrate_(self):
        W = self.weight.data; qmax = 2**self.n_bits - 1
        if self.per_row:
            wmin = W.min(dim=1, keepdim=True).values
            wmax = W.max(dim=1, keepdim=True).values
            scale = (wmax - wmin).clamp_min(1e-8) / qmax
            zp = (-wmin / scale).round().clamp(0, qmax)
            self.scale.copy_(scale.squeeze(1)); self.zp.copy_(zp.squeeze(1))
        else:
            wmin, wmax = W.min(), W.max()
            scale = (wmax - wmin).clamp_min(1e-8) / qmax
            zp = (-wmin / scale).round().clamp(0, qmax)
            self.scale.fill_(scale); self.zp.fill_(zp)
        self.calibrated.fill_(1)

    def _deq_weight(self):
        W = self.weight; qmax = 2**self.n_bits - 1
        if self.per_row:
            scale = self.scale[:, None]; zp = self.zp[:, None]
        else:
            scale = self.scale; zp = self.zp
        Q = (W / scale + zp).round().clamp(0, qmax)
        return (Q - zp) * scale

    def forward(self, input):
        if self.calibrated.item() == 0:
            self.calibrate_()
        return F.embedding(input, self._deq_weight(), self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

def quantize_roberta_embeddings_inplace(qa_model: AutoModelForQuestionAnswering, n_bits=4, per_row=True):
    """
    Replace backbone embeddings with QuantEmbedding (row-wise INTx fake-quant).
    Handles:
      - roberta.embeddings.word_embeddings
      - roberta.embeddings.position_embeddings
      - roberta.embeddings.token_type_embeddings (if present)
    """
    if not hasattr(qa_model, "roberta"):
        raise RuntimeError("This script expects a RoBERTa QA model exposing `.roberta`.")
    emb = qa_model.roberta.embeddings

    def _swap(name: str):
        mod = getattr(emb, name, None)
        if isinstance(mod, nn.Embedding):
            qe = QuantEmbedding(mod.num_embeddings, mod.embedding_dim,
                                padding_idx=mod.padding_idx, max_norm=mod.max_norm,
                                norm_type=mod.norm_type, scale_grad_by_freq=mod.scale_grad_by_freq,
                                sparse=mod.sparse, n_bits=n_bits, per_row=per_row).to(mod.weight.device, mod.weight.dtype)
            with torch.no_grad(): qe.weight.copy_(mod.weight.data)
            setattr(emb, name, qe)

    _swap("word_embeddings")
    _swap("position_embeddings")
    if hasattr(emb, "token_type_embeddings"):
        _swap("token_type_embeddings")

# ------------- Embed-aware size counter (MB) -------------
def compute_quantized_params_embed_aware(model) -> float:
    """
    Count parameters with correct bit-widths:
      - QuantEmbedding.weight -> module.n_bits
      - modules with .weight_quantizer -> that n_bits
      - lora/adapter low-rank weights (if named accordingly) -> 16 bits
      - others -> dtype bits (FP32/FP16/BF16)
    """
    total_bits = 0
    for name, module in model.named_modules():
        params = getattr(module, "_parameters", {})
        for pname, p in params.items():
            if p is None: continue
            if isinstance(module, QuantEmbedding) and pname == "weight":
                n_bits = int(module.n_bits)
            elif (pname == "weight") and hasattr(module, "weight_quantizer"):
                n_bits = int(getattr(module.weight_quantizer, "n_bits", torch.finfo(p.dtype).bits))
            elif "lora" in pname or "adapter" in name:
                n_bits = 16
            else:
                n_bits = torch.finfo(p.dtype).bits if p.is_floating_point() else (p.element_size() * 8)
            total_bits += p.numel() * n_bits
    return total_bits // 8 / 1e6  # MB

# ------------------ Adapters (1D) -------------------
class Gate(nn.Module):
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x): return self.alpha * x

class DepthwiseSeparableAdapter1D(nn.Module):
    """Token-sequence adapter for attention residual (B, N, D)."""
    def __init__(self, dim: int, use_pw: bool=True, r: int=64, act: str="gelu", gate_init: float=0.1):
        super().__init__()
        self.dim = dim; self.use_pw = use_pw
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)
        self.act = nn.GELU() if act=="gelu" else (nn.SiLU() if act=="silu" else nn.Identity())
        if use_pw:
            self.pw1 = nn.Conv1d(dim, r, kernel_size=1, bias=True)
            self.pw2 = nn.Conv1d(r, dim, kernel_size=1, bias=True)
            nn.init.kaiming_uniform_(self.pw1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.pw2.weight)
            nn.init.zeros_(self.pw1.bias); nn.init.zeros_(self.pw2.bias)
        else:
            self.pw1 = self.pw2 = None
        self.gate = Gate(gate_init)
    def forward(self, x):                # (B, N, D)
        B, N, D = x.shape; assert D == self.dim
        t = x.transpose(1, 2).contiguous()   # (B, D, N)
        y = self.dw(t); y = self.act(y)
        if self.use_pw: y = self.pw2(self.act(self.pw1(y)))
        y = y.transpose(1, 2).contiguous()
        return self.gate(y)

class MLPTokenAdapter(nn.Module):
    """Token-wise adapter injected into FFN residual."""
    def __init__(self, dim: int, kind: Literal["affine","lowrank","bottleneck"]="lowrank",
                 r: int=64, act: str="gelu", gate_init: float=0.1):
        super().__init__()
        self.kind = kind
        if kind == "affine":
            self.scale = nn.Parameter(torch.ones(dim))
            self.bias  = nn.Parameter(torch.zeros(dim))
            self.act = nn.Identity(); self.low1=self.low2=self.mid=None
        else:
            self.act = nn.GELU() if act=="gelu" else (nn.SiLU() if act=="silu" else nn.Identity())
            self.low1 = nn.Linear(dim, r, bias=True)
            self.low2 = nn.Linear(r, dim, bias=True)
            nn.init.zeros_(self.low2.weight)
            nn.init.zeros_(self.low1.bias); nn.init.zeros_(self.low2.bias)
            if kind == "bottleneck":
                self.mid = nn.Linear(r, r, bias=True)
                nn.init.kaiming_uniform_(self.low1.weight, a=math.sqrt(5))
                nn.init.zeros_(self.mid.weight); nn.init.zeros_(self.mid.bias)
            else:
                self.mid = None
                nn.init.kaiming_uniform_(self.low1.weight, a=math.sqrt(5))
        self.gate = Gate(gate_init)
    def forward(self, x):
        if self.kind == "affine": y = x * self.scale + self.bias
        elif self.kind == "lowrank": y = self.low2(self.low1(x))
        else: y = self.low2(self.act(self.mid(self.act(self.low1(x)))))
        return self.gate(y)

# -------------- RoBERTa layer wrapper (pre-LN enc) --------------
class WrappedBertLikeLayer(nn.Module):
    """
    Works for RoBERTa (and BERT). Injects adapters on residual inputs.
    Filters kwargs to match underlying HF signatures.
    """
    def __init__(self, layer: nn.Module, hidden_size: int, attn_r=64, mlp_kind="lowrank", mlp_r=64):
        super().__init__()
        self.layer = layer
        self.attn_adapter = DepthwiseSeparableAdapter1D(hidden_size, use_pw=True, r=attn_r)
        self.mlp_adapter  = MLPTokenAdapter(hidden_size, kind=mlp_kind, r=mlp_r)

    def _filter_kwargs(self, fn, cand: dict):
        allowed = set(inspect.signature(fn).parameters.keys())
        return {k: v for k, v in cand.items() if (v is not None and k in allowed)}

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        attention_mask         = kwargs.pop("attention_mask", None)
        head_mask              = kwargs.pop("head_mask", None)
        output_attentions      = kwargs.pop("output_attentions", False)
        use_cache              = kwargs.pop("use_cache", False)
        cache_position         = kwargs.pop("cache_position", None)
        past_key_value         = kwargs.pop("past_key_value", None)
        past_key_values        = kwargs.pop("past_key_values", None)
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values

        # ---- Self-attention ----
        attn_self = self.layer.attention.self
        attn_kwargs = self._filter_kwargs(attn_self.forward, {
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "output_attentions": output_attentions,
            "past_key_value": past_key_value,
            "cache_position": cache_position,
        })
        # past_key_values compatibility:
        if ("past_key_value" not in attn_kwargs) and (past_key_value is not None):
            if "past_key_values" in inspect.signature(attn_self.forward).parameters:
                attn_kwargs["past_key_values"] = past_key_value

        self_outputs = attn_self(hidden_states, **attn_kwargs)
        attn_context = self_outputs[0]  # (B, N, D)

        # Adapter on residual feeding the attention output proj
        a_hat = self.attn_adapter(hidden_states)

        # attention.output(hidden_states, input_tensor)
        attn_out = self.layer.attention.output
        attention_output = attn_out(attn_context, hidden_states + a_hat)

        # ---- FFN ----
        intermediate_output = self.layer.intermediate(attention_output)
        m_hat = self.mlp_adapter(attention_output)
        layer_output = self.layer.output(intermediate_output, attention_output + m_hat)

        outputs = (layer_output,)
        if output_attentions and len(self_outputs) > 1:
            outputs = outputs + (self_outputs[1],)
        if use_cache:
            outputs = outputs + (None,)
        return outputs

def add_layer_adapters(qa_model, attn_r=64, mlp_kind="lowrank", mlp_r=64, freeze_backbone=True):
    ref = next(qa_model.parameters())
    dev, dtype = ref.device, ref.dtype

    bb = qa_model.roberta if hasattr(qa_model, "roberta") else qa_model.bert
    layers = bb.encoder.layer
    hidden_size = qa_model.config.hidden_size

    new_layers, adapter_params = [], []
    for blk in layers:
        wrapped = WrappedBertLikeLayer(blk, hidden_size, attn_r=attn_r, mlp_kind=mlp_kind, mlp_r=mlp_r).to(dev, dtype=dtype)
        new_layers.append(wrapped)
        for m in wrapped.modules():
            if isinstance(m, (DepthwiseSeparableAdapter1D, MLPTokenAdapter, Gate)):
                adapter_params.extend(list(m.parameters()))

    bb.encoder.layer = nn.ModuleList(new_layers)

    if freeze_backbone:
        for p in qa_model.parameters(): p.requires_grad_(False)
    # Deduplicate and mark adapter params trainable
    seen, uniq = set(), []
    for p in adapter_params:
        if id(p) not in seen:
            seen.add(id(p)); p.requires_grad_(True); uniq.append(p)
    return uniq

# ---------------- SQuAD v1.1 prep ----------------
def prepare_features(tokenizer, examples, max_len=384, doc_stride=128, pad_on_right=True, is_train=True):
    questions = [q.lstrip() for q in examples["question"]]
    contexts  = examples["context"]

    tokenized = tokenizer(
        questions if pad_on_right else contexts,
        contexts if pad_on_right else questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    features = {k: tokenized[k] for k in tokenized}

    if is_train:
        start_positions = []; end_positions = []
        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]
            start_char = answers["answer_start"][0]
            end_char   = start_char + len(answers["text"][0])
            sequence_ids = tokenized.sequence_ids(i)
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != (1 if pad_on_right else 0): idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == (1 if pad_on_right else 0): idx += 1
            context_end = idx - 1
            if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
                start_positions.append(0); end_positions.append(0); continue
            start_token = context_start
            while start_token <= context_end and offsets[start_token][0] <= start_char: start_token += 1
            start_token -= 1
            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= end_char: end_token -= 1
            end_token += 1
            start_positions.append(start_token); end_positions.append(end_token)
        features["start_positions"] = start_positions
        features["end_positions"] = end_positions
    else:
        example_ids = []
        for i in range(len(offset_mapping)):
            sample_idx = sample_mapping[i]
            example_ids.append(examples["id"][sample_idx])
            sequence_ids = tokenized.sequence_ids(i)
            offsets = offset_mapping[i]
            token_offsets = []
            for k, off in enumerate(offsets):
                if sequence_ids[k] == (1 if pad_on_right else 0): token_offsets.append(off)
                else: token_offsets.append((0,0))
            offset_mapping[i] = token_offsets
        features["example_id"] = example_ids
    features["offset_mapping"] = offset_mapping
    return features

def build_dataloaders(tokenizer, max_len=384, doc_stride=128, batch_size=16, num_workers=8):
    raw = load_dataset("squad")  # v1.1
    pad_on_right = True

    train_proc = raw["train"].map(
        lambda ex: prepare_features(tokenizer, ex, max_len, doc_stride, pad_on_right, is_train=True),
        batched=True, remove_columns=raw["train"].column_names, desc="map train"
    )
    train_proc.set_format(type="torch")

    val_examples = raw["validation"]
    val_features_full = val_examples.map(
        lambda ex: prepare_features(tokenizer, ex, max_len, doc_stride, pad_on_right, is_train=False),
        batched=True, remove_columns=val_examples.column_names, desc="map validation"
    )

    val_features_tensors = val_features_full.remove_columns(
        [c for c in val_features_full.column_names if c not in ("input_ids","attention_mask")]
    )
    val_features_tensors.set_format(type="torch")

    train_loader = torch.utils.data.DataLoader(
        train_proc, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=default_data_collator, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_features_tensors, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=default_data_collator, pin_memory=True
    )
    return raw, train_proc, val_examples, val_features_full, train_loader, val_loader

# --------------- Post-process + evaluation ---------------
def postprocess_squad1_fast(examples, features, start_logits, end_logits,
                            n_best=20, max_answer_len=30, device="cuda"):
    ex_ids = list(examples["id"])
    id_to_row = {eid: i for i, eid in enumerate(ex_ids)}
    example_to_features = {}
    for i, ex_id in enumerate(features["example_id"]):
        example_to_features.setdefault(ex_id, []).append(i)

    if isinstance(start_logits, np.ndarray):
        start_t = torch.from_numpy(start_logits)
        end_t   = torch.from_numpy(end_logits)
    else:
        start_t, end_t = start_logits, end_logits
    start_t = start_t.to(device, non_blocking=True)
    end_t   = end_t.to(device, non_blocking=True)

    predictions = {}
    for ex_id, feat_inds in example_to_features.items():
        ctx = examples["context"][id_to_row[ex_id]]
        best_score = -float("inf"); best_text = ""
        for i in feat_inds:
            s = start_t[i]; e = end_t[i]
            offsets = np.array(features["offset_mapping"][i], dtype=np.int32)
            valid   = (offsets[:,0] != 0) | (offsets[:,1] != 0)

            L = s.numel()
            idx = torch.arange(L, device=device)
            start_idx = idx.view(-1, 1); end_idx = idx.view(1, -1)

            valid_t = torch.from_numpy(valid).to(device)
            valid_mask = valid_t.view(-1,1) & valid_t.view(1,-1)
            tri_mask   = end_idx >= start_idx
            len_mask   = (end_idx - start_idx + 1) <= max_answer_len
            mask = valid_mask & tri_mask & len_mask

            scores = s.view(-1,1) + e.view(1,-1)
            scores = scores.masked_fill(~mask, float("-inf"))

            flat_idx = torch.argmax(scores)
            if torch.isneginf(scores.view(-1)[flat_idx]): continue
            start_best = (flat_idx // L).item(); end_best = (flat_idx % L).item()

            start_char, end_char = offsets[start_best, 0], offsets[end_best, 1]
            if end_char > start_char:
                text = ctx[start_char:end_char]
                score = (s[start_best] + e[end_best]).item()
                if score > best_score: best_score, best_text = score, text

        predictions[ex_id] = best_text if best_text is not None else ""
    return predictions

@torch.no_grad()
def evaluate_qa(model, tokenizer, val_examples, val_features_full, val_loader, device):
    model.eval()
    start_logits_all, end_logits_all = [], []
    from contextlib import nullcontext
    cm = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with cm:
        for batch in val_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                     if k in ("input_ids","attention_mask")}
            out = model(**batch, return_dict=True)
            start_logits_all.append(out.start_logits.detach())
            end_logits_all.append(out.end_logits.detach())
    start_logits = torch.cat(start_logits_all, dim=0)
    end_logits   = torch.cat(end_logits_all,   dim=0)

    preds = postprocess_squad1_fast(
        examples=val_examples, features=val_features_full,
        start_logits=start_logits, end_logits=end_logits,
        n_best=20, max_answer_len=30, device=device.type
    )
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_examples]
    metric = evaluate.load("squad")
    results = metric.compute(
        predictions=[{"id": k, "prediction_text": v} for k, v in preds.items()],
        references=references
    )
    return results

# -------------------- RepQ calibration --------------------
@torch.no_grad()
def repq_calibrate(model, loader, device, steps=500):
    set_quant_state(model, input_quant=True, weight_quant=True)
    model.eval()
    it = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ("input_ids","attention_mask","start_positions","end_positions")}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True)
        it += 1
        if it >= steps: break

# --------------------- KD + CE loss -----------------------
class QA_KD_CE(nn.Module):
    def __init__(self, tau=2.0, alpha_kd=0.5, alpha_ce=0.5):
        super().__init__()
        self.tau=tau; self.alpha_kd=alpha_kd; self.alpha_ce=alpha_ce
        self.ce = nn.CrossEntropyLoss()
    def forward(self, s_start, s_end, t_start, t_end, y_start, y_end):
        ce = self.ce(s_start, y_start) + self.ce(s_end, y_end)
        T = self.tau
        kd = (T*T) * (
            F.kl_div(F.log_softmax(s_start/T, dim=-1), F.softmax(t_start/T, dim=-1), reduction="batchmean") +
            F.kl_div(F.log_softmax(s_end/T,   dim=-1), F.softmax(t_end/T,   dim=-1), reduction="batchmean")
        )
        return self.alpha_ce * ce + self.alpha_kd * kd

# ----------------- Train adapters only --------------------
def train_adapters(model_q, teacher_fp32, train_loader, val_examples, val_features_full, val_loader, tokenizer,
                   device, epochs=1, lr=2e-4, weight_decay=0.01, grad_accum=1, max_grad_norm=1.0,
                   trainable_params: List[nn.Parameter] = None):
    model_q.train(); teacher_fp32.eval()
    assert trainable_params and len(trainable_params) > 0, "No trainable adapter params!"
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    crit = QA_KD_CE(tau=2.0, alpha_kd=0.5, alpha_ce=0.5)

    best = {"exact_match": 0.0, "f1": 0.0}
    step = 0
    for ep in range(1, epochs+1):
        run_loss = 0.0; n_steps = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids","attention_mask","start_positions","end_positions")}
            with torch.no_grad():
                tout = teacher_fp32(**{k:batch[k] for k in ("input_ids","attention_mask")}, return_dict=True)
                t_start, t_end = tout.start_logits, tout.end_logits
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                sout = model_q(**{k:batch[k] for k in ("input_ids","attention_mask")}, return_dict=True)
                s_start, s_end = sout.start_logits, sout.end_logits
                loss = crit(s_start, s_end, t_start, t_end,
                            batch["start_positions"], batch["end_positions"]) / grad_accum
            scaler.scale(loss).backward()
            if (step+1) % grad_accum == 0:
                if max_grad_norm is not None:
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1; n_steps += 1; run_loss += loss.item() * grad_accum
            if step % 100 == 0:
                print(f"[epoch {ep} step {step}] loss={run_loss/max(1,n_steps):.4f}")

        eval_res = evaluate_qa(model_q, tokenizer, val_examples, val_features_full, val_loader, device)
        print(f"[epoch {ep}] SQuAD v1.1 dev: EM={eval_res['exact_match']:.2f} F1={eval_res['f1']:.2f}")
        if eval_res["f1"] > best["f1"]:
            best = eval_res
            torch.save(model_q.state_dict(), "roberta_repq_adapters_best.pth")
            print("  ✅ New best saved: roberta_repq_adapters_best.pth")
    return best

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: use a RoBERTa QA checkpoint (SQuAD v1.1 fine-tuned)
    # You can change this if you prefer another RoBERTa QA model.
    ap.add_argument("--teacher_id", type=str, default="csarron/roberta-base-squad-v1")
    ap.add_argument("--student_init", type=str, default=None, help="defaults to teacher_id if None")

    # Data/loader
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)

    # RepQ PTQ
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--a_bits", type=int, default=8)
    ap.add_argument("--calib_steps", type=int, default=400)

    # Embedding quant
    ap.add_argument("--embed_bits", type=int, default=8)
    ap.add_argument("--embed_per_row", action="store_true", default=True)

    # Training (adapters)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Adapters
    ap.add_argument("--attn_r", type=int, default=8)
    ap.add_argument("--mlp_r", type=int, default=8)
    ap.add_argument("--mlp_kind", type=str, default="lowrank", choices=["affine","lowrank","bottleneck"])

    # Optional logs/size counter
    ap.add_argument("--local_rank", type=int, default=0)
    ap.add_argument("--log_file", type=str, default="log.txt")

    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    # --- Tokenizer ---
    teacher_id = args.teacher_id
    student_id = args.student_init if args.student_init else teacher_id
    tokenizer = AutoTokenizer.from_pretrained(student_id, use_fast=True)
    if tokenizer.pad_token is None:
        # RoBERTa usually has <pad>, but just in case:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    # --- Data ---
    raw, train_proc, val_examples, val_features_full, train_loader, val_loader = build_dataloaders(
        tokenizer, max_len=args.max_len, doc_stride=args.doc_stride,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # --- Teacher FP32 ---
    t_cfg = AutoConfig.from_pretrained(teacher_id)
    t_cfg.pad_token_id = tokenizer.pad_token_id
    teacher = AutoModelForQuestionAnswering.from_pretrained(teacher_id, config=t_cfg).to(device).eval()
    fp32_eval = evaluate_qa(teacher, tokenizer, val_examples, val_features_full, val_loader, device)
    print(f"[FP32 baseline] SQuAD v1.1 dev: EM={fp32_eval['exact_match']:.2f} F1={fp32_eval['f1']:.2f}")

    # --- Student FP32 init (clone arch) ---
    s_cfg = AutoConfig.from_pretrained(student_id)
    s_cfg.pad_token_id = tokenizer.pad_token_id
    student_fp32 = AutoModelForQuestionAnswering.from_pretrained(student_id, config=s_cfg).to(device).eval()

    # Optional legacy size counter
    try:
        size_fp32_std = compute_quantized_params(student_fp32, local_rank=args.local_rank, log_file=args.log_file)
        print(f"FP32 size (legacy counter): {size_fp32_std:.3f} MB")
    except Exception as e:
        print("[warn] compute_quantized_params failed:", e)

    # --- RepQ wrap (weight+act) ---
    q_model = quant_model(
        student_fp32,
        input_quant_params={'n_bits': args.a_bits, 'channel_wise': False},
        weight_quant_params={'n_bits': args.w_bits, 'channel_wise': True}
    ).to(device).eval()
    qmods = [m for m in q_model.modules() if hasattr(m, "weight_quantizer")]
    print("Quantized modules (RepQ-wrapped):", len(qmods))

    # --- Quantize embeddings with per-row INTx (on the final q_model) ---
    quantize_roberta_embeddings_inplace(q_model, n_bits=args.embed_bits, per_row=args.embed_per_row)
    # Calibrate QuantEmbedding once (weights) before RepQ act calib
    for m in q_model.modules():
        if isinstance(m, QuantEmbedding) and m.calibrated.item() == 0:
            m.calibrate_()

    # --- RepQ activation calibration ---
    print("Calibrating RepQ on SQuAD train...")
    repq_calibrate(q_model, train_loader, device, steps=args.calib_steps)
    print("Calibration done.")

    # Size counters
    try:
        size_ptq_std = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
        print(f"Size after RepQ (legacy counter): {size_ptq_std:.3f} MB")
    except Exception as e:
        print("[warn] compute_quantized_params failed:", e)
    size_embed_aware = compute_quantized_params_embed_aware(q_model)
    print(f"Size after RepQ (embed-aware): {size_embed_aware:.3f} MB")

    # --- PTQ baseline eval ---
    repq_eval = evaluate_qa(q_model, tokenizer, val_examples, val_features_full, val_loader, device)
    print(f"[RepQ PTQ baseline] SQuAD v1.1 dev: EM={repq_eval['exact_match']:.2f} F1={repq_eval['f1']:.2f}")

    # --- Insert adapters & train adapters only ---
    adapter_params = add_layer_adapters(
        q_model, attn_r=args.attn_r, mlp_kind=args.mlp_kind, mlp_r=args.mlp_r, freeze_backbone=True
    )
    q_model.to(device)
    size_embed_aware_adp = compute_quantized_params_embed_aware(q_model)
    print(f"Size (embed-aware + adapters): {size_embed_aware_adp:.3f} MB "
          f"(adapters add ≈ {size_embed_aware_adp - size_embed_aware:.3f} MB)")
    print(f"Adapter-only params: {sum(p.numel() for p in adapter_params)/1e6:.2f}M")

    best = train_adapters(
        q_model, teacher, train_loader,
        val_examples, val_features_full, val_loader, tokenizer,
        device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm, trainable_params=adapter_params
    )

    # --- Final eval ---
    final_eval = evaluate_qa(q_model, tokenizer, val_examples, val_features_full, val_loader, device)
    print("\n=== Final Comparison (SQuAD v1.1 dev) ===")
    print(f"FP32 baseline:       EM={fp32_eval['exact_match']:.2f} F1={fp32_eval['f1']:.2f}")
    print(f"RepQ PTQ baseline:   EM={repq_eval['exact_match']:.2f} F1={repq_eval['f1']:.2f}")
    print(f"RepQ + Adapters:     EM={final_eval['exact_match']:.2f} F1={final_eval['f1']:.2f}")
    print("=========================================")

if __name__ == "__main__":
    main()
