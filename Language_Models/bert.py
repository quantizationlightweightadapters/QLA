#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT-base on MRPC with RepQ PTQ + ViT-style parallel adapters per block

- Keeps your RepQ flow
- Embeddings are wrapped with QuantEmbedding (fake-quant weights)
- Adapters unchanged
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math, random, argparse
from typing import Dict, Any, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

# ---- RepQ (your local library) ----
from quant import quant_model, set_quant_state, QuantLinear, QuantMatMul  # noqa: F401
from utils.utils import write, compute_quantized_params as _compute_quantized_params_fallback

# ================================
# Utilities
# ================================

def seed_everything(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

def collate_fn_builder(tokenizer, max_len=128):
    def collate(exs):
        s1 = [e["sentence1"] for e in exs]
        s2 = [e["sentence2"] for e in exs]
        labels = torch.tensor([e["label"] for e in exs], dtype=torch.long)
        enc = tokenizer(
            s1, s2,
            padding="max_length", truncation=True, max_length=max_len,
            return_tensors="pt"
        )
        enc["labels"] = labels
        return enc
    return collate

# ================================
# Fake-quantized Embedding (weights)
# ================================

class QuantEmbedding(nn.Embedding):
    """
    Fake-quantize the embedding WEIGHTS (per-row affine, INT4 by default).
    - Stores FP32 weights as usual (for training/ease).
    - During forward, uses dequantized weights Q_deq = dequant(quantize(W)).
    - scale/zp computed once via calibrate_() and cached as buffers.
    """
    def __init__(self, num_embeddings, embedding_dim, *, n_bits: int = 4, per_row: bool = True, **kw):
        super().__init__(num_embeddings, embedding_dim, **kw)
        self.n_bits = int(n_bits)
        self.per_row = bool(per_row)
        # Buffers for static quant params
        if per_row:
            self.register_buffer("scale", torch.ones(num_embeddings), persistent=True)
            self.register_buffer("zp",    torch.zeros(num_embeddings), persistent=True)
        else:
            self.register_buffer("scale", torch.ones(1), persistent=True)
            self.register_buffer("zp",    torch.zeros(1), persistent=True)
        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.uint8), persistent=True)

    @torch.no_grad()
    def calibrate_(self):
        W = self.weight.data
        nb = self.n_bits
        qmax = 2**nb - 1
        if self.per_row:
            # per-row min/max (row = vocabulary entry)
            wmin = W.min(dim=1, keepdim=True).values
            wmax = W.max(dim=1, keepdim=True).values
            scale = (wmax - wmin).clamp_min(1e-8) / qmax
            zp = (-wmin / scale).round().clamp(0, qmax)
            self.scale.copy_(scale.squeeze(1))
            self.zp.copy_(zp.squeeze(1))
        else:
            wmin, wmax = W.min(), W.max()
            scale = (wmax - wmin).clamp_min(1e-8) / qmax
            zp = (-wmin / scale).round().clamp(0, qmax)
            self.scale.fill_(scale)
            self.zp.fill_(zp)
        self.calibrated.fill_(1)

    def _dequantized_weight(self):
        # Quantize-dequantize weights on-the-fly (fake quant)
        W = self.weight
        nb = self.n_bits
        qmax = 2**nb - 1
        if self.per_row:
            scale = self.scale[:, None]
            zp = self.zp[:, None]
        else:
            scale = self.scale
            zp = self.zp
        Q = (W / scale + zp).round().clamp(0, qmax)          # int domain
        W_deq = (Q - zp) * scale                              # dequant back to float
        return W_deq

    def forward(self, input):
        if self.calibrated.item() == 0:
            # Self-calibrate on first use if user forgot to call calibrate_()
            self.calibrate_()
        W_deq = self._dequantized_weight()
        return F.embedding(input, W_deq, self.padding_idx, self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.sparse)

    @staticmethod
    def from_embedding(emb: nn.Embedding, n_bits=4, per_row=True) -> "QuantEmbedding":
        qe = QuantEmbedding(
            num_embeddings=emb.num_embeddings,
            embedding_dim=emb.embedding_dim,
            padding_idx=emb.padding_idx,
            max_norm=emb.max_norm,
            norm_type=emb.norm_type,
            scale_grad_by_freq=emb.scale_grad_by_freq,
            sparse=emb.sparse,
            _weight=None,
            n_bits=n_bits,
            per_row=per_row,
        )
        with torch.no_grad():
            qe.weight.copy_(emb.weight.data)
        return qe

def apply_quant_embeddings_to_bert_sc_model(sc_model: AutoModelForSequenceClassification,
                                            n_bits: int = 4,
                                            per_row: bool = True):
    """
    Replace BERT embeddings with QuantEmbedding:
      - word_embeddings
      - position_embeddings
      - token_type_embeddings
    """
    if not hasattr(sc_model, "bert"):
        return

    emb_mod = sc_model.bert.embeddings

    # Word embeddings
    if isinstance(emb_mod.word_embeddings, nn.Embedding):
        new_we = QuantEmbedding.from_embedding(emb_mod.word_embeddings, n_bits=n_bits, per_row=per_row)
        new_we.to(device=emb_mod.word_embeddings.weight.device, dtype=emb_mod.word_embeddings.weight.dtype)
        emb_mod.word_embeddings = new_we

    # Position embeddings
    if isinstance(emb_mod.position_embeddings, nn.Embedding):
        new_pe = QuantEmbedding.from_embedding(emb_mod.position_embeddings, n_bits=n_bits, per_row=per_row)
        new_pe.to(device=emb_mod.position_embeddings.weight.device, dtype=emb_mod.position_embeddings.weight.dtype)
        emb_mod.position_embeddings = new_pe

    # Token-type embeddings (may be absent in some models/configs)
    if hasattr(emb_mod, "token_type_embeddings") and isinstance(emb_mod.token_type_embeddings, nn.Embedding):
        new_te = QuantEmbedding.from_embedding(emb_mod.token_type_embeddings, n_bits=n_bits, per_row=per_row)
        new_te.to(device=emb_mod.token_type_embeddings.weight.device, dtype=emb_mod.token_type_embeddings.weight.dtype)
        emb_mod.token_type_embeddings = new_te

# ================================
# Adapters (unchanged from your code)
# ================================

class Gate(nn.Module):
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x): return self.alpha * x

class DepthwiseSeparableAdapter1D(nn.Module):
    def __init__(self, dim: int, use_pw: bool = True, r: int = 32, act: str = "gelu", gate_init: float = 0.1):
        super().__init__()
        self.dim = dim
        self.use_pw = use_pw
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)
        self.act = nn.GELU() if act == "gelu" else (nn.SiLU() if act == "silu" else nn.Identity())
        if use_pw:
            self.pw1 = nn.Conv1d(dim, r, kernel_size=1, bias=True)
            self.pw2 = nn.Conv1d(r, dim, kernel_size=1, bias=True)
            nn.init.kaiming_uniform_(self.pw1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.pw2.weight); nn.init.zeros_(self.pw1.bias); nn.init.zeros_(self.pw2.bias)
        else:
            self.pw1 = self.pw2 = None
        self.gate = Gate(gate_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_cls, x_tok = x[:, :1, :], x[:, 1:, :]
        t = x_tok.transpose(1, 2).contiguous()
        y = self.dw(t); y = self.act(y)
        if self.use_pw:
            y = self.pw2(self.act(self.pw1(y)))
        y = y.transpose(1, 2).contiguous()
        y_full = torch.cat([x_cls, y], dim=1)
        return self.gate(y_full)

class MLPTokenAdapter(nn.Module):
    def __init__(self, dim: int, kind: Literal["affine","lowrank","bottleneck"]="affine",
                 r: int = 32, act: str = "gelu", gate_init: float = 0.1):
        super().__init__()
        self.dim = dim; self.kind = kind
        if kind == "affine":
            self.scale = nn.Parameter(torch.ones(dim))
            self.bias  = nn.Parameter(torch.zeros(dim))
            self.act = nn.Identity(); self.low1 = self.low2 = self.mid = None
        else:
            self.act = nn.GELU() if act == "gelu" else (nn.SiLU() if act == "silu" else nn.Identity())
            self.low1 = nn.Linear(dim, r, bias=True)
            self.low2 = nn.Linear(r, dim, bias=True)
            nn.init.zeros_(self.low2.weight); nn.init.zeros_(self.low1.bias); nn.init.zeros_(self.low2.bias)
            if kind == "bottleneck":
                self.mid = nn.Linear(r, r, bias=True)
                nn.init.kaiming_uniform_(self.low1.weight, a=math.sqrt(5))
                nn.init.zeros_(self.mid.weight); nn.init.zeros_(self.mid.bias)
            else:
                self.mid = None
                nn.init.kaiming_uniform_(self.low1.weight, a=math.sqrt(5))
        self.gate = Gate(gate_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "affine":
            y = x * self.scale + self.bias
        elif self.kind == "lowrank":
            y = self.low2(self.low1(x))
        else:
            y = self.low2(self.act(self.mid(self.act(self.low1(x)))))
        return self.gate(y)

class WrappedBertLayer(nn.Module):
    def __init__(self, bert_layer: nn.Module, dim: int,
                 attn_use_pw: bool=True, attn_r: int=32, attn_act: str="gelu", attn_gate: float=0.1,
                 mlp_kind: str="affine", mlp_r: int=32, mlp_act: str="gelu", mlp_gate: float=0.1):
        super().__init__()
        self.layer = bert_layer
        self.dim = dim
        self.attn_adapter = DepthwiseSeparableAdapter1D(dim, use_pw=attn_use_pw, r=attn_r, act=attn_act, gate_init=attn_gate)
        self.mlp_adapter  = MLPTokenAdapter(dim, kind=mlp_kind, r=mlp_r, act=mlp_act, gate_init=mlp_gate)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, output_attentions=False, **kwargs):
        u = hidden_states
        attn_outs = self.layer.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions
        )
        attention_output = attn_outs[0]
        a_hat = self.attn_adapter(u)
        x = attention_output + a_hat

        v = x
        intermediate_output = self.layer.intermediate(v)
        ffn_out = self.layer.output.dense(intermediate_output)
        ffn_out = self.layer.output.dropout(ffn_out)
        m_hat = self.mlp_adapter(v)
        ffn_out = ffn_out + m_hat
        layer_output = self.layer.output.LayerNorm(ffn_out + v)

        outputs = (layer_output,)
        if output_attentions:
            outputs = outputs + attn_outs[1:]
        return outputs

def add_bert_compensation_adapters(sc_model: AutoModelForSequenceClassification,
                                   attn_use_pw=True, attn_r=64,
                                   mlp_kind="lowrank", mlp_r=64,
                                   freeze_backbone=True):
    bert = sc_model.bert
    hidden_size = bert.config.hidden_size
    enc = bert.encoder

    ref_param = next(sc_model.parameters())
    param_device = ref_param.device
    param_dtype  = ref_param.dtype

    new_layers = []
    adapter_params: list[torch.nn.Parameter] = []
    for layer in enc.layer:
        wrapped = WrappedBertLayer(
            layer, hidden_size,
            attn_use_pw=attn_use_pw, attn_r=attn_r, attn_act="gelu", attn_gate=0.1,
            mlp_kind=mlp_kind, mlp_r=mlp_r, mlp_act="gelu", mlp_gate=0.1
        )
        wrapped.to(device=param_device, dtype=param_dtype)
        new_layers.append(wrapped)
        for m in wrapped.modules():
            if isinstance(m, (DepthwiseSeparableAdapter1D, MLPTokenAdapter, Gate)):
                for p in m.parameters():
                    adapter_params.append(p)
    enc.layer = nn.ModuleList(new_layers)

    if freeze_backbone:
        for p in sc_model.parameters():
            p.requires_grad_(False)
    for p in adapter_params:
        p.requires_grad_(True)
    return adapter_params

# ================================
# Loss + Eval
# ================================

class KDCECriterion(nn.Module):
    def __init__(self, tau=4.0, alpha_kd=0.6, alpha_ce=0.4):
        super().__init__()
        self.tau = tau; self.alpha_kd = alpha_kd; self.alpha_ce = alpha_ce
        self.ce  = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits_s, logits_t, labels):
        loss_ce = self.ce(logits_s, labels)
        T = self.tau
        log_p_s = F.log_softmax(logits_s / T, dim=-1)
        p_t     = F.softmax(logits_t / T, dim=-1)
        loss_kd = (T*T) * self.kld(log_p_s, p_t)
        loss = self.alpha_ce * loss_ce + self.alpha_kd * loss_kd
        return loss, {"loss": float(loss.detach()), "ce": float(loss_ce.detach()),
                      "kd": float(loss_kd.detach())}

@torch.no_grad()
def evaluate_sc_model(sc_model: nn.Module, loader, device) -> Dict[str, Any]:
    sc_model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = sc_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            return_dict=False
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out.logits
        ps.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(batch["labels"].cpu().tolist())
    return {"acc": accuracy_score(ys, ps), "f1": f1_score(ys, ps)}

# ================================
# RepQ quant + calibration
# ================================

def repq_build_quant(student_fp32: AutoModelForSequenceClassification,
                     w_bits=8, a_bits=8,
                     embed_bits=4, embed_per_row=True) -> nn.Module:
    """
    1) Wrap embeddings with QuantEmbedding (fake-quant).
    2) Apply your quant_model() to the rest (it typically skips embeddings).
    """
    # 1) Make embeddings fake-quant
    apply_quant_embeddings_to_bert_sc_model(student_fp32, n_bits=embed_bits, per_row=embed_per_row)

    # 2) Quantize linear/matmul etc. via your RepQ rewrite
    wq_params = {'n_bits': w_bits, 'channel_wise': True}
    aq_params = {'n_bits': a_bits, 'channel_wise': False}
    q_model = quant_model(student_fp32, input_quant_params=aq_params, weight_quant_params=wq_params)
    return q_model

@torch.no_grad()
def repq_calibrate(model: nn.Module, loader, device, steps: int = 200):
    """
    Enable quantizers and run a few batches to collect min/max.
    Also calibrate QuantEmbedding (static weight ranges).
    """
    # Turn on quantization for RepQ modules
    set_quant_state(model, input_quant=True, weight_quant=True)

    # Calibrate QuantEmbedding once (weight-only)
    for m in model.modules():
        if isinstance(m, QuantEmbedding) and m.calibrated.item() == 0:
            m.calibrate_()

    # Optionally run a few batches to calibrate activation quantizers etc.
    model.eval()
    it = 0
    for batch in loader:
        batch = to_device(batch, device)
        _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            return_dict=False
        )
        it += 1
        if it >= steps:
            break

# ================================
# Train loop (adapters only)
# ================================

def train_adapters_model(
    teacher_sc: AutoModelForSequenceClassification,
    student_sc: AutoModelForSequenceClassification,  # quantized + adapters inside
    train_loader,
    val_loader,
    device: torch.device,
    epochs=3,
    lr=2e-4,
    weight_decay=0.01,
    tau=4.0,
    alpha_kd=0.6,
    alpha_ce=0.4,
    grad_accum=1,
    max_grad_norm=1.0,
    trainable_params=None,
):
    teacher_sc.eval()
    student_sc.train()

    if trainable_params is None:
        trainable_params = []
        for m in student_sc.modules():
            if isinstance(m, (DepthwiseSeparableAdapter1D, MLPTokenAdapter, Gate)):
                for p in m.parameters():
                    if p.requires_grad:
                        trainable_params.append(p)

    assert len(trainable_params) > 0, "No trainable adapter parameters found."

    optim = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    crit = KDCECriterion(tau=tau, alpha_kd=alpha_kd, alpha_ce=alpha_ce)

    best = {"acc": 0.0, "f1": 0.0}
    for epoch in range(1, epochs + 1):
        student_sc.train()
        run = {"loss":0.0,"ce":0.0,"kd":0.0,"n":0}

        for i, batch in enumerate(train_loader, 1):
            batch = to_device(batch, device)

            with torch.no_grad():
                t_out = teacher_sc(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids", None),
                    return_dict=True
                )
                logits_t = t_out.logits

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                s_out = student_sc(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids", None),
                    return_dict=False
                )
                logits_s = s_out[0] if isinstance(s_out, (tuple, list)) else s_out.logits
                loss, logs = crit(logits_s=logits_s, logits_t=logits_t, labels=batch["labels"])
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            if i % grad_accum == 0:
                if max_grad_norm is not None:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            run["loss"] += logs["loss"]; run["ce"] += logs["ce"]; run["kd"] += logs["kd"]; run["n"] += 1
            if i % 50 == 0:
                avg_loss = run["loss"]/max(1,run["n"]); avg_ce = run["ce"]/max(1,run["n"]); avg_kd = run["kd"]/max(1,run["n"])
                print(f"[Epoch {epoch} | step {i}] loss={avg_loss:.4f} ce={avg_ce:.4f} kd={avg_kd:.4f}")

        eval_metrics = evaluate_sc_model(student_sc, val_loader, device)
        print(f"[Epoch {epoch}] MRPC val: acc={eval_metrics['acc']:.4f} f1={eval_metrics['f1']:.4f}")

        if eval_metrics["f1"] > best["f1"]:
            best = eval_metrics
            torch.save(student_sc.state_dict(), "bert_repq_block_adapters_best.pth")
            print("  ✅ New best saved -> bert_repq_block_adapters_best.pth")

    print(f"Best after adapters: acc={best['acc']:.4f} f1={best['f1']:.4f}")
    return best

# ================================
# Size reporting
# ================================

def compute_quantized_params(model, local_rank=0, log_file=None):
    """
    Like yours, but counts QuantEmbedding.weight at its n_bits.
    Falls back to your original helper for everything else.
    """
    quantized_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantEmbedding):
            p = module.weight
            if p is not None:
                n_bits = getattr(module, "n_bits", torch.finfo(p.dtype).bits)
                quantized_bits += p.numel() * n_bits

    # Add everything else using your original helper (assumes it counts RepQ weights correctly)
    total_mb = _compute_quantized_params_fallback(model, local_rank=local_rank, log_file=log_file)
    # Replace the FP32 accounting of embeddings inside fallback with our int4 count:
    # We can't easily subtract, so instead: recompute from scratch simply here:

    # Full recompute simple path:
    total_bits = 0
    for _name_, _module_ in model.named_modules():
        if len(getattr(_module_, "_parameters", {})) > 0:
            for k in _module_._parameters:
                p = _module_._parameters[k]
                if p is None:
                    continue
                if isinstance(_module_, QuantEmbedding) and k == "weight":
                    n_bits_ = _module_.n_bits
                elif (k == 'weight') and hasattr(_module_, 'weight_quantizer'):
                    n_bits_ = _module_.weight_quantizer.n_bits
                elif 'lora_weight' in k:
                    n_bits_ = 16
                else:
                    n_bits_ = torch.finfo(p.dtype).bits if p.is_floating_point() else (p.element_size()*8)
                total_bits += p.numel() * n_bits_
    return total_bits // 8 / 1e6  # MB

# ================================
# Main
# ================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--teacher_ckpt", type=str, default="textattack/bert-base-uncased-MRPC")
    ap.add_argument("--student_init", type=str, default=None)
    ap.add_argument("--use_safetensors", action="store_true", default=True)

    # RepQ PTQ
    ap.add_argument("--w_bits", type=int, default=6)
    ap.add_argument("--a_bits", type=int, default=6)
    ap.add_argument("--embed_bits", type=int, default=6)
    ap.add_argument("--embed_per_row", action="store_true", default=True)
    ap.add_argument("--calib_steps", type=int, default=200)

    # Data
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    # Train
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
    print(f"Device: {device}")

    # Data
    raw = load_dataset("glue", "mrpc")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_ckpt, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    collate = collate_fn_builder(tokenizer, max_len=args.max_len)
    train_loader = torch.utils.data.DataLoader(raw["train"], batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               collate_fn=collate, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(raw["validation"], batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers,
                                               collate_fn=collate, pin_memory=True)

    # Teacher FP32
    t_cfg = AutoConfig.from_pretrained(args.teacher_ckpt, num_labels=2)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_ckpt, config=t_cfg, use_safetensors=args.use_safetensors
    ).to(device).eval()

    # Student FP32 init
    student_id = args.student_init if args.student_init else args.teacher_ckpt
    s_cfg = AutoConfig.from_pretrained(student_id, num_labels=2)
    s_cfg.return_dict = False
    student_fp32 = AutoModelForSequenceClassification.from_pretrained(
        student_id, config=s_cfg, use_safetensors=args.use_safetensors
    ).to(device).eval()

    # FP32 baseline
    fp32_metrics = evaluate_sc_model(student_fp32, val_loader, device)
    print(f"[FP32 baseline] MRPC val: acc={fp32_metrics['acc']:.4f} f1={fp32_metrics['f1']:.4f}")
    
    f32_params_mb = compute_quantized_params(student_fp32, local_rank=args.local_rank, log_file=args.log_file)
    print('F32 model size estimate: {:.3f} MB'.format(f32_params_mb))

    # ---- RepQ PTQ (fake-quant) ----
    q_model = repq_build_quant(
        student_fp32, w_bits=args.w_bits, a_bits=args.a_bits,
        embed_bits=args.embed_bits, embed_per_row=args.embed_per_row
    ).to(device).eval()

    print("Calibrating RepQ encodings (incl. QuantEmbedding)…")
    repq_calibrate(q_model, train_loader, device, steps=args.calib_steps)
    print("Calibration done.")
    set_quant_state(q_model, input_quant=True, weight_quant=True)

    # PTQ baseline
    ptq_metrics = evaluate_sc_model(q_model, val_loader, device)
    print(f"[RepQ-PTQ baseline] MRPC val: acc={ptq_metrics['acc']:.4f} f1={ptq_metrics['f1']:.4f}")

    ptq_params_mb = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    print('RepQ-PTQ size estimate: {:.3f} MB'.format(ptq_params_mb))

    # Inject adapters INTO the quantized model
    adapter_params = add_bert_compensation_adapters(
        q_model,
        attn_use_pw=True, attn_r=args.attn_r,
        mlp_kind=args.mlp_kind, mlp_r=args.mlp_r,
        freeze_backbone=True
    )
    q_model.to(device)

    # Sanity check before adapter training
    pre_metrics = evaluate_sc_model(q_model, val_loader, device)
    print(f"[RepQ-PTQ + Adapters (init)] MRPC val: acc={pre_metrics['acc']:.4f} f1={pre_metrics['f1']:.4f}")

    ptq_params_mb = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    print('Model + adapters size estimate: {:.3f} MB'.format(ptq_params_mb))

    # Train adapters (KD + CE)
    best = train_adapters_model(
        teacher_sc=teacher,
        student_sc=q_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tau=4.0,
        alpha_kd=0.6,
        alpha_ce=0.4,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        trainable_params=adapter_params
    )

    # Final metrics
    final_metrics = evaluate_sc_model(q_model, val_loader, device)
    print(f"[RepQ-PTQ + Adapters (trained)] MRPC val: acc={final_metrics['acc']:.4f} f1={final_metrics['f1']:.4f}")

    # Summary
    print("\n=== Final Comparison (MRPC Validation) ===")
    print(f"FP32 Model:                        acc={fp32_metrics['acc']:.4f}, f1={fp32_metrics['f1']:.4f}")
    print(f"RepQ Quantized:    acc={ptq_metrics['acc']:.4f}, f1={ptq_metrics['f1']:.4f}")
    print(f"RepQ + Adapters:     acc={final_metrics['acc']:.4f}, f1={final_metrics['f1']:.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
