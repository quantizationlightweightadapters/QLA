#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 (decoder-only) on SQuAD v1.1 with RepQ PTQ + ViT-style block adapters

- Model: GPT2ForQuestionAnswering (use a pretrained SQuAD-v1.1 checkpoint)
- Data: SQuAD v1.1 (train/dev)
- Quant: RepQ (fake quant) with calibration on SQuAD train
- Adapters: DepthwiseSeparableAdapter1D (attn branch) + MLPTokenAdapter (mlp branch), parallel add (pre-LN)
- Train: freeze backbone (quantized), optimize ONLY adapters with KD + CE (start/end)
- Metrics: EM/F1 on SQuAD v1.1 dev (HF evaluate)

Install:
  pip install transformers datasets evaluate safetensors scikit-learn

RepQ:
  Your repo must expose:
    from quant.quant_model import quant_model, set_quant_state
"""

import os, sys, math, random, argparse
from typing import Optional, Tuple, List, Literal, Dict, Any

# --- Make parent dir importable so we can import quant/quant_model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quant.quant_model import quant_model, set_quant_state  # <-- RepQ
from utils import *
from utils.utils import write, create_transform, create_loader, AverageMeter, broadcast_tensor_from_main_process, gather_tensor_from_multi_processes, compute_quantized_params
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
    GPT2ForQuestionAnswering,
    default_data_collator,
)
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ----------------------- Repro -----------------------
def seed_everything(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ------------------ Adapters (1D) -------------------
class Gate(nn.Module):
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x): return self.alpha * x

class DepthwiseSeparableAdapter1D(nn.Module):
    """
    Token-sequence adapter for decoder blocks (no CLS):
    x: (B, N, D) → Conv1d over N (groups=D), + optional PW bottleneck D->r->D, Gate.
    """
    def __init__(self, dim: int, use_pw: bool=True, r: int=64, act: str="gelu", gate_init: float=0.1):
        super().__init__()
        self.dim = dim
        self.use_pw = use_pw
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
        B, N, D = x.shape
        assert D == self.dim
        t = x.transpose(1, 2).contiguous()   # (B, D, N)
        y = self.dw(t)
        y = self.act(y)
        if self.use_pw:
            y = self.pw2(self.act(self.pw1(y)))
        y = y.transpose(1, 2).contiguous()   # (B, N, D)
        return self.gate(y)

class MLPTokenAdapter(nn.Module):
    """
    Token-wise adapter on MLP branch:
       affine | lowrank (D->r->D) | bottleneck (D->r->r->D)
    """
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
        if self.kind == "affine":
            y = x * self.scale + self.bias
        elif self.kind == "lowrank":
            y = self.low2(self.low1(x))
        else:
            y = self.low2(self.act(self.mid(self.act(self.low1(x)))))
        return self.gate(y)

# -------- Wrap GPT-2 blocks with parallel adapters --------
class WrappedGPT2Block(nn.Module):
    """
    HF GPT2Block (pre-LN):
      u = ln_1(x); attn(u) -> add to x
      v = ln_2(x); mlp(v)  -> add to x
    Inject parallel adapters computed from u, v.
    """
    def __init__(self, gpt2_block: nn.Module, hidden_size: int,
                 attn_use_pw=True, attn_r=64, mlp_kind="lowrank", mlp_r=64):
        super().__init__()
        self.block = gpt2_block
        self.attn_adapter = DepthwiseSeparableAdapter1D(hidden_size, use_pw=attn_use_pw, r=attn_r)
        self.mlp_adapter  = MLPTokenAdapter(hidden_size, kind=mlp_kind, r=mlp_r)

    def forward(self, hidden_states: torch.FloatTensor, *args, **kwargs):
        # Pull args from kwargs safely (works across HF versions)
        layer_past = kwargs.get("layer_past", None)
        attention_mask = kwargs.get("attention_mask", None)
        head_mask = kwargs.get("head_mask", None)
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        use_cache = kwargs.get("use_cache", False)
        output_attentions = kwargs.get("output_attentions", False)

        # ---- attn branch (pre-LN) ----
        u = self.block.ln_1(hidden_states)
        attn_outputs = self.block.attn(
            u,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # (B,N,D)
        a_hat = self.attn_adapter(u)
        x = hidden_states + attn_output + a_hat

        # ---- mlp branch (pre-LN) ----
        v = self.block.ln_2(x)
        mlp_out = self.block.mlp(v)
        m_hat = self.mlp_adapter(v)
        x = x + mlp_out + m_hat

        outputs = (x,)
        if use_cache:
            # past_key_value is at index 1 in HF GPT2Block outputs
            outputs = outputs + (attn_outputs[1],)
        if output_attentions:
            outputs = outputs + (attn_outputs[-1],)
        return outputs

def add_gpt2_block_adapters(qamodel: GPT2ForQuestionAnswering,
                            attn_r=64, mlp_kind="lowrank", mlp_r=64,
                            freeze_backbone=True) -> List[nn.Parameter]:
    tr = qamodel.transformer
    hs = tr.config.hidden_size
    ref_param = next(qamodel.parameters())
    dev, dtype = ref_param.device, ref_param.dtype

    new_layers = []
    adapter_params: List[nn.Parameter] = []
    for blk in tr.h:
        wrapped = WrappedGPT2Block(blk, hs, attn_use_pw=True, attn_r=attn_r,
                                   mlp_kind=mlp_kind, mlp_r=mlp_r).to(dev, dtype=dtype)
        new_layers.append(wrapped)
        for m in wrapped.modules():
            if isinstance(m, (DepthwiseSeparableAdapter1D, MLPTokenAdapter, Gate)):
                for p in m.parameters():
                    adapter_params.append(p)
    tr.h = nn.ModuleList(new_layers)

    if freeze_backbone:
        for p in qamodel.parameters():
            p.requires_grad_(False)
    for p in adapter_params:
        p.requires_grad_(True)

    return adapter_params

# ------------------- SQuAD v1.1 prep -------------------
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
        start_positions = []
        end_positions = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = features["input_ids"][i]
            # Map back to original example
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]
            # SQuAD v1.1 has at least one answer
            start_char = answers["answer_start"][0]
            end_char   = start_char + len(answers["text"][0])

            sequence_ids = tokenized.sequence_ids(i)
            # Find context token range in this feature
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != (1 if pad_on_right else 0): idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == (1 if pad_on_right else 0): idx += 1
            context_end = idx - 1

            # If answer not fully inside this feature, set to cls-like anchor (we'll use position 0)
            if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
                start_positions.append(0); end_positions.append(0); continue

            # Otherwise locate start/end tokens
            start_token = context_start
            while start_token <= context_end and offsets[start_token][0] <= start_char: start_token += 1
            start_token -= 1

            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= end_char: end_token -= 1
            end_token += 1

            start_positions.append(start_token)
            end_positions.append(end_token)

        features["start_positions"] = start_positions
        features["end_positions"] = end_positions
    else:
        # Validation: keep example_id & filtered offset mapping
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

def build_dataloaders(tokenizer, max_len=384, doc_stride=128, batch_size=8, num_workers=0):
    raw = load_dataset("squad")  # v1.1
    # raw["train"] = raw["train"].select(range(5000))
    # raw["validation"] = raw["validation"].select(range(1000))
    pad_on_right = True

    # Train features
    train_proc = raw["train"].map(
        lambda ex: prepare_features(tokenizer, ex, max_len, doc_stride, pad_on_right, is_train=True),
        batched=True, remove_columns=raw["train"].column_names, desc="map train"
    )
    train_proc.set_format(type="torch")

    # Validation features
    val_examples = raw["validation"]
    val_features_full = val_examples.map(
        lambda ex: prepare_features(tokenizer, ex, max_len, doc_stride, pad_on_right, is_train=False),
        batched=True, remove_columns=val_examples.column_names, desc="map validation"
    )
    # Tensors for loader (keep only ids/masks)
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
def postprocess_squad1(tokenizer, examples, features, raw_start_logits, raw_end_logits,
                       n_best=20, max_answer_len=30):
    from collections import defaultdict
    example_to_features = defaultdict(list)
    for i, ex_id in enumerate(features["example_id"]):
        example_to_features[ex_id].append(i)

    predictions = {}
    for ex_id, feat_inds in example_to_features.items():
        context = examples["context"][examples["id"].index(ex_id)]
        prelim = []
        for i in feat_inds:
            start_logit = raw_start_logits[i]
            end_logit   = raw_end_logits[i]
            offsets     = features["offset_mapping"][i]
            input_ids   = features["input_ids"][i]
            # top-k indices
            start_indexes = np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
            end_indexes   = np.argsort(end_logit)[-1:-n_best-1:-1].tolist()
            for s in start_indexes:
                for e in end_indexes:
                    if s >= len(offsets) or e >= len(offsets): continue
                    if offsets[s] == (0,0) or offsets[e] == (0,0): continue
                    if e < s or (e - s + 1) > max_answer_len: continue
                    score = start_logit[s] + end_logit[e]
                    prelim.append({"score": float(score), "start": s, "end": e,
                                   "offsets": (offsets[s][0], offsets[e][1])})
        if prelim:
            best = max(prelim, key=lambda x: x["score"])
            start_char, end_char = best["offsets"]
            text = context[start_char:end_char]
        else:
            text = ""
        predictions[ex_id] = text
    return predictions

@torch.no_grad()
def evaluate_qa(model, tokenizer, val_examples, val_features_full, val_loader, device):
    model.eval()
    all_start, all_end = [], []
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids","attention_mask")}
        out = model(**batch, return_dict=True)
        all_start.extend(out.start_logits.detach().cpu().split(1, dim=0))
        all_end.extend(out.end_logits.detach().cpu().split(1, dim=0))
    # keep per-feature arrays aligned with val_features_full
    start_logits = [x.squeeze(0).numpy() for x in all_start]
    end_logits   = [x.squeeze(0).numpy() for x in all_end]

    preds = postprocess_squad1(tokenizer, val_examples, val_features_full, start_logits, end_logits)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_examples]
    metric = evaluate.load("squad")
    results = metric.compute(predictions=[{"id": k, "prediction_text": v} for k,v in preds.items()],
                             references=references)
    return results  # {'exact_match':..., 'f1':...}

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
        loss = self.alpha_ce * ce + self.alpha_kd * kd
        return loss, {"loss": float(loss.detach()), "ce": float(ce.detach()), "kd": float(kd.detach())}

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
        run = {"loss":0.0,"ce":0.0,"kd":0.0,"n":0}
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids","attention_mask","start_positions","end_positions")}
            with torch.no_grad():
                tout = teacher_fp32(**{k:batch[k] for k in ("input_ids","attention_mask")}, return_dict=True)
                t_start, t_end = tout.start_logits, tout.end_logits
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                sout = model_q(**{k:batch[k] for k in ("input_ids","attention_mask")}, return_dict=True)
                s_start, s_end = sout.start_logits, sout.end_logits
                loss, logs = crit(s_start, s_end, t_start, t_end,
                                  batch["start_positions"], batch["end_positions"])
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (step+1) % grad_accum == 0:
                if max_grad_norm is not None:
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            run["loss"] += logs["loss"]; run["ce"] += logs["ce"]; run["kd"] += logs["kd"]; run["n"] += 1
            if step % 100 == 0:
                avg = {k: run[k]/max(1,run["n"]) for k in ("loss","ce","kd")}
                print(f"[epoch {ep} step {step}] loss={avg['loss']:.4f} ce={avg['ce']:.4f} kd={avg['kd']:.4f}")

        eval_res = evaluate_qa(model_q, tokenizer, val_examples, val_features_full, val_loader, device)
        print(f"[epoch {ep}] SQuAD v1.1 dev: EM={eval_res['exact_match']:.2f} F1={eval_res['f1']:.2f}")
        if eval_res["f1"] > best["f1"]:
            best = eval_res
            torch.save(model_q.state_dict(), "gpt2_repq_adapters_best.pth")
            print("  ✅ New best saved: gpt2_repq_adapters_best.pth")
    return best

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Model
    ap.add_argument("--teacher_ckpt", type=str, default="varun-v-rao/gpt2-large-squad-model3",
                    help="Any GPT-2 QA checkpoint for SQuAD v1.1 (or 'gpt2' then fine-tune yourself)")
    ap.add_argument("--student_init", type=str, default=None)

    # Data/loader
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)

    # RepQ PTQ
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--a_bits", type=int, default=8)
    ap.add_argument("--calib_steps", type=int, default=400)

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
    ap.add_argument("--local_rank", type=int, default=0)
    ap.add_argument('--log_file', type=str, default='log.txt')

    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    # Tokenizer (GPT-2: no pad by default → set pad to eos)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_ckpt, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Data
    raw, train_proc, val_examples, val_features_full, train_loader, val_loader = build_dataloaders(
        tokenizer, max_len=args.max_len, doc_stride=args.doc_stride,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Teacher FP32
    t_cfg = AutoConfig.from_pretrained(args.teacher_ckpt)
    t_cfg.pad_token_id = tokenizer.pad_token_id
    teacher = GPT2ForQuestionAnswering.from_pretrained(args.teacher_ckpt, config=t_cfg).to(device).eval()

    # FP32 baseline
    fp32_eval = evaluate_qa(teacher, tokenizer, val_examples, val_features_full, val_loader, device)
    print(f"[FP32 baseline] SQuAD v1.1 dev: EM={fp32_eval['exact_match']:.2f} F1={fp32_eval['f1']:.2f}")
    
    total_f32, trainable_f32 = count_params(teacher)
    print(f"[FP32 baseline]: total={total_f32/1e6:.2f}M, trainable={trainable_f32/1e6:.2f}M")

    # Student FP32 init (copy of teacher arch)
    student_id = args.student_init if args.student_init else args.teacher_ckpt
    s_cfg = AutoConfig.from_pretrained(student_id)
    s_cfg.pad_token_id = tokenizer.pad_token_id
    student_fp32 = GPT2ForQuestionAnswering.from_pretrained(student_id, config=s_cfg).to(device).eval()

    # RepQ quantize
    q_model = quant_model(
        student_fp32,
        input_quant_params={'n_bits': args.a_bits, 'channel_wise': False},
        weight_quant_params={'n_bits': args.w_bits, 'channel_wise': True}
    ).to(device).eval()

    print("Calibrating RepQ on SQuAD train...")
    repq_calibrate(q_model, train_loader, device, steps=args.calib_steps)
    print("Calibration done.")

    total_before, trainable_before = count_params(q_model)
    print(f"Before adapters: total={total_before/1e6:.2f}M, trainable={trainable_before/1e6:.2f}M")

    # PTQ baseline
    repq_eval = evaluate_qa(q_model, tokenizer, val_examples, val_features_full, val_loader, device)
    print(f"[RepQ PTQ baseline] SQuAD v1.1 dev: EM={repq_eval['exact_match']:.2f} F1={repq_eval['f1']:.2f}")
    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    print('RepQ-ViT model with adapters size is {:.3f}'.format(ptq_params))
    # Insert adapters (freeze backbone) & train adapters only
    adapter_params = add_gpt2_block_adapters(q_model,
                                             attn_r=args.attn_r,
                                             mlp_kind=args.mlp_kind,
                                             mlp_r=args.mlp_r,
                                             freeze_backbone=True)
    q_model.to(device)
    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    print('RepQ-ViT model with adapters size is {:.3f}'.format(ptq_params))
    best = train_adapters(q_model, teacher, train_loader,
                          val_examples, val_features_full, val_loader, tokenizer,
                          device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                          grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm, trainable_params=adapter_params)
    total_after, trainable_after = count_params(q_model)
    print(f"After adapters: total={total_after/1e6:.2f}M, trainable={trainable_after/1e6:.2f}M")

    print(f"Adapter-only params: {sum(p.numel() for p in adapter_params)/1e6:.2f}M")
    # Final eval
    final_eval = evaluate_qa(q_model, tokenizer, val_examples, val_features_full, val_loader, device)
    print("\n=== Final Comparison (SQuAD v1.1 dev) ===")
    print(f"FP32 baseline:       EM={fp32_eval['exact_match']:.2f} F1={fp32_eval['f1']:.2f}")
    print(f"RepQ PTQ baseline:   EM={repq_eval['exact_match']:.2f} F1={repq_eval['f1']:.2f}")
    print(f"RepQ + Adapters:     EM={final_eval['exact_match']:.2f} F1={final_eval['f1']:.2f}")
    print("=========================================")

if __name__ == "__main__":
    main()
