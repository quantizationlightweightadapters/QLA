#!/usr/bin/env python3
""" Fair comparison script:
- Build SAME quantized baseline as their code (RepQ-ViT PTQ + reparam)
- Replace their per-block linear "CompensationBlock" with YOUR adapters
- Train only adapters with KD + CE (optionally feature matching)
- Evaluate with their validate() pipeline (now robust to logits-or-tuple)

Original credits retained where applicable.
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import copy
import random
import socket
from contextlib import suppress
from functools import partial
from typing import Optional, Sequence, Literal, Dict, Tuple, List
from tqdm import tqdm

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler

import timm
from timm.data.dataset import ImageDataset

# --- Their utility stack (as in your provided script) ---
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import random_seed, NativeScaler, accuracy as timm_accuracy
from timm.scheduler.scheduler_factory import CosineLRScheduler

import sys
from quant import *
from utils import *
from utils.utils import write, create_transform, create_loader, AverageMeter, broadcast_tensor_from_main_process, gather_tensor_from_multi_processes, compute_quantized_params

HOST_NAME = socket.getfqdn(socket.gethostname())
torch.backends.cudnn.benchmark = True

# ---------------------------
# Your Adapter & Trainer code
# ---------------------------

class Gate(nn.Module):
    """Learnable scalar gate; init small so adapter starts near zero."""
    def __init__(self, init=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x): return self.alpha * x

class DepthwiseSeparableAdapter(nn.Module):
    """
    Spatial adapter on patch grid: DW 3x3 (+ optional PW bottleneck D->r->D), residual-style.
    Expects input of shape (B, N, D), with N = 1 + H*W (CLS + patches).
    CLS is bypassed around the spatial conv.
    """
    def __init__(self, dim: int, use_pw: bool = True, r: int = 32, act: str = "gelu", gate_init: float = 0.1):
        super().__init__()
        self.dim = dim
        self.use_pw = use_pw
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()

        if use_pw:
            self.pw1 = nn.Conv2d(dim, r, kernel_size=1, bias=True)
            self.pw2 = nn.Conv2d(r, dim, kernel_size=1, bias=True)
            # init: first is small, last is zero so whole adapter starts ~0
            nn.init.kaiming_uniform_(self.pw1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.pw2.weight)
            nn.init.zeros_(self.pw1.bias); nn.init.zeros_(self.pw2.bias)
        else:
            self.pw1 = self.pw2 = None

        self.gate = Gate(gate_init)

    @staticmethod
    def _tokens_to_grid(x_tok: torch.Tensor):
        # x_tok: (B, 196, D) -> (B, D, 14, 14) for 224/16 ViTs
        B, Nt, D = x_tok.shape
        H = W = int(math.sqrt(Nt))
        assert H * W == Nt, f"Token count {Nt} not a perfect square."
        return x_tok.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _grid_to_tokens(x_grid: torch.Tensor):
        # x_grid: (B, D, 14, 14) -> (B, 196, D)
        B, D, H, W = x_grid.shape
        return x_grid.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)  (CLS + patches), returns adapter output with same shape for residual add.
        """
        B, N, D = x.shape
        assert D == self.dim, f"Adapter dim {self.dim} vs input {D}"

        x_cls = x[:, :1, :]            # (B, 1, D)
        x_tok = x[:, 1:, :]            # (B, 196, D) for 224/16
        x_grid = self._tokens_to_grid(x_tok)   # (B, D, 14, 14)
        y = self.dw(x_grid)
        y = self.act(y)
        if self.use_pw:
            y = self.pw2(self.act(self.pw1(y)))

        y_tok = self._grid_to_tokens(y)       # (B, 196, D)
        y_full = torch.cat([x_cls, y_tok], dim=1)  # (B, N, D)
        return self.gate(y_full)              # scaled residual contribution


class MLPTokenAdapter(nn.Module):
    """
    Token-wise adapter for MLP branch. Choices:
      - 'affine': per-channel scale & bias (foldable), very tiny.
      - 'lowrank': linear D->r->D (no nonlinearity), zero-init last.
      - 'bottleneck': D->r->r->D with GELU, zero-init last.
    """
    def __init__(self, dim: int, kind: Literal["affine","lowrank","bottleneck"]="affine",
                 r: int = 32, act: str = "gelu", gate_init: float = 0.1):
        super().__init__()
        self.dim = dim
        self.kind = kind

        if kind == "affine":
            self.scale = nn.Parameter(torch.ones(dim))
            self.bias  = nn.Parameter(torch.zeros(dim))
            self.act = nn.Identity()
            self.low1 = self.low2 = None

        else:
            if act == "gelu":
                self.act = nn.GELU()
            elif act == "silu":
                self.act = nn.SiLU()
            else:
                self.act = nn.Identity()

            self.low1 = nn.Linear(dim, r, bias=True)
            self.low2 = nn.Linear(r, dim, bias=True)
            nn.init.zeros_(self.low2.weight)   # zero-init last so adapter starts as no-op
            nn.init.zeros_(self.low1.bias); nn.init.zeros_(self.low2.bias)

            if kind == "bottleneck":
                self.mid = nn.Linear(r, r, bias=True)
                nn.init.kaiming_uniform_(self.low1.weight, a=math.sqrt(5))
                nn.init.zeros_(self.mid.weight); nn.init.zeros_(self.mid.bias)
            else:
                self.mid = None
                nn.init.kaiming_uniform_(self.low1.weight, a=math.sqrt(5))

        self.gate = Gate(gate_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)  (CLS included)
        if self.kind == "affine":
            y = x * self.scale + self.bias
        elif self.kind == "lowrank":
            y = self.low2(self.low1(x))
        else:  # bottleneck
            y = self.low2(self.act(self.mid(self.act(self.low1(x)))))
        return self.gate(y)


class WrappedViTBlock(nn.Module):
    """
    Wrap a timm ViT Block (pre-LN) and add:
      - Parallel DW(+PW) adapter on the Attention sublayer (input = norm1(x), add at same residual)
      - Parallel token-wise adapter on the MLP sublayer (input = norm2(x), add at same residual)
    """
    def __init__(self, block, dim: int,
                 attn_use_pw: bool=True, attn_r: int=32, attn_act: str="gelu", attn_gate: float=0.1,
                 mlp_kind: Literal["affine","lowrank","bottleneck"]="affine", mlp_r: int=32, mlp_act: str="gelu", mlp_gate: float=0.1):
        super().__init__()
        self.block = block
        self.dim = dim
        self.attn_adapter = DepthwiseSeparableAdapter(dim, use_pw=attn_use_pw, r=attn_r, act=attn_act, gate_init=attn_gate)
        self.mlp_adapter  = MLPTokenAdapter(dim, kind=mlp_kind, r=mlp_r, act=mlp_act, gate_init=mlp_gate)
        self.has_dp = hasattr(block, 'drop_path') and isinstance(block.drop_path, nn.Module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer
        u = self.block.norm1(x)
        a = self.block.attn(u)
        a_hat = self.attn_adapter(u)
        a_sum = a + a_hat
        x = x + (self.block.drop_path(a_sum) if self.has_dp else a_sum)

        # MLP sublayer
        v = self.block.norm2(x)
        m = self.block.mlp(v)
        m_hat = self.mlp_adapter(v)
        m_sum = m + m_hat
        x = x + (self.block.drop_path(m_sum) if self.has_dp else m_sum)
        return x


class ViTQuantWithAdapters(nn.Module):
    """
    Wrap an quantized timm ViT with parallel adapters on chosen blocks.
    """
    def __init__(self, quant_vit: nn.Module,
                 blocks_to_adapt: Optional[Sequence[int]] = None,
                 attn_use_pw: bool=True, attn_r: int=32, attn_act: str="gelu", attn_gate: float=0.1,
                 mlp_kind: Literal["affine","lowrank","bottleneck"]="affine", mlp_r: int=32, mlp_act: str="gelu", mlp_gate: float=0.1,
                 freeze_backbone: bool=True):
        super().__init__()
        self.backbone = quant_vit
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        D = self.backbone.blocks[0].norm1.normalized_shape[0]
        n_blocks = len(self.backbone.blocks)
        if blocks_to_adapt is None or blocks_to_adapt == "all":
            blocks_to_adapt = list(range(n_blocks))
        self.blocks_to_adapt = set(blocks_to_adapt)

        wrapped = []
        for i, blk in enumerate(self.backbone.blocks):
            if i in self.blocks_to_adapt:
                wrapped.append(
                    WrappedViTBlock(
                        blk, D,
                        attn_use_pw=attn_use_pw, attn_r=attn_r, attn_act=attn_act, attn_gate=attn_gate,
                        mlp_kind=mlp_kind, mlp_r=mlp_r, mlp_act=mlp_act, mlp_gate=mlp_gate
                    )
                )
            else:
                wrapped.append(blk)
        self.backbone.blocks = nn.Sequential(*wrapped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def adapters_parameters(self):
        """Iterator over ONLY adapter parameters (for optimizer)."""
        for m in self.modules():
            if isinstance(m, (DepthwiseSeparableAdapter, MLPTokenAdapter, Gate)):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p


def add_vit_compensation_adapters(sim_model: nn.Module,
                                  blocks: Optional[Sequence[int]] = None,
                                  attn_use_pw=True, attn_r=64,
                                  mlp_kind="affine", mlp_r=64,
                                  freeze_backbone=True) -> ViTQuantWithAdapters:
    return ViTQuantWithAdapters(
        quant_vit=sim_model,
        blocks_to_adapt=blocks if blocks is not None else "all",
        attn_use_pw=attn_use_pw, attn_r=attn_r,
        mlp_kind=mlp_kind, mlp_r=mlp_r,
        freeze_backbone=freeze_backbone
    )

def _get_blocks_seq(model: nn.Module) -> nn.Sequential:
    m = model
    if hasattr(m, "module"): m = m.module
    if hasattr(m, "backbone"): m = m.backbone
    if not hasattr(m, "blocks"):
        raise AttributeError(f"{type(m).__name__} has no attribute 'blocks'")
    return m.blocks

def register_block_output_hooks(vit_like_model: nn.Module, block_indices, detach=True):
    blocks = _get_blocks_seq(vit_like_model)
    buf, handles = {}, []
    for i in block_indices:
        def _make_hook(i_):
            def hook(mod, inp, out):
                buf[i_] = out.detach() if detach else out
            return hook
        handles.append(blocks[i].register_forward_hook(_make_hook(i)))
    return handles, buf

def remove_hooks(handles):
    for h in handles: h.remove()

class DistillLoss(nn.Module):
    def __init__(self, tau=2.0, alpha_kd=0.5, alpha_fm=1e-3):
        super().__init__()
        self.tau = tau
        self.alpha_kd = alpha_kd
        self.alpha_fm = alpha_fm
        self.ce = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction="batchmean")
    def forward(self, logits_s, logits_t, labels, feats_s, feats_t):
        loss_ce = self.ce(logits_s, labels)
        t = self.tau
        log_p_s = F.log_softmax(logits_s / t, dim=1)
        p_t     = F.softmax(logits_t / t, dim=1)
        loss_kd = (t * t) * self.kld(log_p_s, p_t)
        loss_fm = 0.0
        if feats_s and feats_t:
            for i in feats_s.keys():
                if i in feats_t:
                    loss_fm = loss_fm + F.mse_loss(feats_s[i], feats_t[i])
        return loss_ce + self.alpha_kd * loss_kd + self.alpha_fm * loss_fm, {
            "ce": loss_ce.detach(),
            "kd": loss_kd.detach(),
            "fm": (loss_fm.detach() if isinstance(loss_fm, torch.Tensor) else torch.tensor(0.0))
        }

def set_trainable_adapters_only(model):
    for p in model.parameters(): p.requires_grad_(False)
    for n, m in model.named_modules():
        if any(k in n for k in ("attn_adapter", "mlp_adapter", "gate")):
            for p in m.parameters(): p.requires_grad_(True)

def assert_adapters_trainable(model):
    bad = [n for n,p in model.named_parameters()
           if ("attn_adapter" in n or "mlp_adapter" in n or "gate" in n) and not p.requires_grad]
    assert not bad, f"Adapter params unexpectedly frozen: {bad}"

def evaluate_top1(model, val_loader, device="cuda", amp=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(amp and device=="cuda")):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[-1]
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def train_adapters(
    student,
    teacher,
    train_loader,
    val_loader=None,
    trainable_params=None,
    device="cuda",
    epochs=10,
    lr=5e-4,
    weight_decay=0.0,
    amp=True,
    tap_blocks=(0,1,2,3,4,5,6,7,8,9,10,11),
    tau=2.0,
    alpha_kd=0.5,
    alpha_fm=1e-3,
    grad_accum=1,
    max_norm=None,
    save_prefix="vit_quant_adapted",
    num_images=40000
):
    student.to(device).train()
    teacher.to(device).eval()
    set_trainable_adapters_only(student)
    assert_adapters_trainable(student)
    if trainable_params is None:
        trainable_params = list(student.adapters_parameters())
    assert len(trainable_params) > 0, "No adapter params to train."

    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    use_amp = (device == 'cuda' or (hasattr(device, "type") and device.type == "cuda"))
    scaler = GradScaler('cuda', enabled=use_amp)
    crit = DistillLoss(tau=tau, alpha_kd=alpha_kd, alpha_fm=alpha_fm)

    s_hooks, s_buf = register_block_output_hooks(student, tap_blocks, detach=False)
    t_hooks, t_buf = register_block_output_hooks(teacher, tap_blocks, detach=True)

    best_acc, best_state = 0.0, None
    try:
        step = 0
        for epoch in range(epochs):
            student.train()
            epoch_step = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast('cuda', enabled=use_amp):
                    s_buf.clear(); t_buf.clear()
                    with torch.no_grad():
                        logits_t = teacher(images)
                    logits_s = student(images)
                    if isinstance(logits_t, tuple): logits_t = logits_t[-1]
                    if isinstance(logits_s, tuple): logits_s = logits_s[-1]
                    loss, _ = crit(logits_s, logits_t, labels, s_buf, t_buf)
                    loss = loss / grad_accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step+1) % grad_accum == 0:
                    if max_norm:
                        if scaler.is_enabled(): scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)
                    if scaler.is_enabled():
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                epoch_step += 1

                if epoch_step > num_images // train_loader.batch_size:
                    _write(f'Breaking from training loop after {str(epoch_step)} steps')
                    break

            if val_loader is not None:
                acc = evaluate_top1(student, val_loader, device=device, amp=amp)
                _write(f"[Adapters] Epoch {epoch+1}: val top1 = {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(student.state_dict())
                    torch.save(best_state, f"{save_prefix}_best.pth")
                    _write(f"  ✅ New best, saved to {save_prefix}_best.pth")
    finally:
        remove_hooks(s_hooks); remove_hooks(t_hooks)

    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    return student, best_acc

# ---------------------------
# Their original script body (kept as much as possible)
# ---------------------------

LINEAR_COMPENSATION_SAMPLES = 512

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def enable_quant(submodel):
    for _, module in submodel.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMatMul)):
            module.set_quant_state(True, True)

def disable_quant(submodel):
    for _, module in submodel.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMatMul)):
            module.set_quant_state(False, False)

class FeatureDataset(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, item): return self.X[item]

# (Their linear_regression left here but not used in the adapters path)
def linear_regression(X, Y, block_id=0):
    X = X.reshape(-1, X.size(-1))
    X = gather_tensor_from_multi_processes(X, world_size=args.world_size)
    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))
    Y = gather_tensor_from_multi_processes(Y, world_size=args.world_size)
    _write('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))
    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y
    W = W_overall[:-1, :]
    b = W_overall[-1, :]
    Y_pred = X @ W + b
    abs_loss = (Y - Y_pred).abs().mean()
    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot
    _write('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))
    return W, b, r2_score

# NOTE: We DO NOT use their CompensationBlock nor generate_compensation_model in the fair-comparison path.
# They’re kept here for completeness but unused.
class CompensationBlock(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True, local_rank=0, block_id=None):
        super().__init__()
        self.block = block
        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            if local_rank == 0:
                _write('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                _write('block {} using lora init'.format(block_id))
    def forward(self, x):
        out = self.block(x)
        if self.training:
            lora_weight = self.lora_weight.float()
            out = out + x @ lora_weight + self.lora_bias
        else:
            lora_weight = self.lora_weight.half()
            out = out + (x.half() @ lora_weight).float() + self.lora_bias
        return out

def generate_compensation_model_unused(*args, **kwargs):
    raise RuntimeError("We are not using the linear CompensationBlock path in this fair-comparison script.")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="deit_tiny",
                    choices=['vit_small', 'vit_base',
                             'deit_tiny', 'deit_small', 'deit_base',
                             'deit_tiny_distilled', 'deit_small_distilled', 'deit_base_distilled'],
                    help="model")
parser.add_argument('--data_dir', default='/opt/Dataset/ImageNet', type=str)

parser.add_argument('--w_bits', default=4, type=int, help='bit-precision of weights')
parser.add_argument('--a_bits', default=4, type=int, help='bit-precision of activation')
parser.add_argument('--blocks', nargs='+', help='Blocks to add QLA layers to', default=None)

parser.add_argument("--batch_size", default=16, type=int, help="batchsize")
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument("--local-rank", default=0, type=int)

# Adapters hyperparams (you can tweak from CLI if you want)
parser.add_argument("--attn_r", default=8, type=int)
parser.add_argument("--mlp_r", default=8, type=int)
parser.add_argument("--mlp_kind", default="lowrank", choices=["affine","lowrank","bottleneck"])
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--num_images", default=40000, type=int)

args = parser.parse_args()

train_aug = 'large_scale_train'
test_aug = 'large_scale_test'
args.drop_path = 0.0
args.num_classes = 1000

model_type = args.model.split("_")[0]
if model_type == "deit":
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225); crop_pct = 0.875
elif model_type == 'vit':
    mean = (0.5, 0.5, 0.5); std = (0.5, 0.5, 0.5); crop_pct = 0.9
elif model_type == 'swin':
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225); crop_pct = 0.9
else:
    raise NotImplementedError

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
args.device = 'cuda:0'
args.world_size = 1
args.rank = 0  # global rank
if args.distributed:
    args.device = f'cuda:{args.local_rank}'
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

assert args.rank >= 0

args.log_dir = os.path.join('checkpoint', args.model, 'FAIR', f'bs_{args.batch_size}_worldsize_{args.world_size}_w_{args.w_bits}_a_{args.a_bits}_sed_{args.seed}')
args.log_file = os.path.join(args.log_dir, 'log.txt')

if args.local_rank == 0:
    os.makedirs(args.log_dir, exist_ok=True)
    if os.path.isfile(args.log_file):
        os.remove(args.log_file)

torch.cuda.synchronize()
_write = partial(write, log_file=args.log_file)

if args.distributed:
    _write('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
else:
    _write('Training with a single process on 1 GPUs.')
assert args.rank >= 0

def validate(model, loader):
    """Robust to model returning logits OR (feat, logits)."""
    top1_m = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda(); target = target.cuda()
            out = model(input)
            if isinstance(out, tuple):
                _, output = out
            else:
                output = out
            acc1, _ = timm_accuracy(output, target, topk=(1, 5))
            top1_m.update(acc1.item(), output.size(0))
        top1_m.synchronize()
    _write('Test  Smples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m))
    return top1_m

def main():
    if args.local_rank == 0:
        _write(vars(args))
    seed_everything(args.seed)
    if args.local_rank == 0:
        _write('dataset mean : {} & std : {}'.format(mean, std))

    # Build datasets/loaders (their utils)
    dataset_train = ImageDataset(root=os.path.join(args.data_dir, 'train'), transform=create_transform(train_aug, mean, std, crop_pct))
    dataset_eval  = ImageDataset(root=os.path.join(args.data_dir, 'val'),   transform=create_transform(test_aug,  mean, std, crop_pct))
    if args.local_rank == 0:
        _write('len of train_set : {}    train_transform : {}'.format(len(dataset_train), dataset_train.transform))
        _write('len of eval_set : {}    eval_transform : {}'.format(len(dataset_eval), dataset_eval.transform))

    loader_train = create_loader(
        dataset_train,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=0.0,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=True,
        local_rank=args.local_rank,
        persistent_workers=False
    )
    
    loader_eval = create_loader(
        dataset_eval,
        batch_size=args.batch_size,
        is_training=False,
        re_prob=0.,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=False,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    # Calibration batch
    for data, target in loader_train:
        calib_data = data.to(args.device)
        break
    broadcast_tensor_from_main_process(calib_data, args)
    _write('local_rank : {} calib_data shape : {} value : {}'.format(args.local_rank, calib_data.size(), calib_data[0, 0, 0, :5]))

    # Model zoo
    model_zoo = {
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'deit_tiny' : 'deit_tiny_patch16_224',
        "deit_tiny_distilled" : "deit_tiny_distilled_patch16_224",
        'deit_small': 'deit_small_patch16_224',
        "deit_small_distilled": "deit_small_distilled_patch16_224",
        'deit_base' : 'deit_base_patch16_224',
        "deit_base_distilled": "deit_base_distilled_patch16_224",
    }

    # Build FP32 model (their function)
    _write('Building model ...')
    model = build_model(model_zoo[args.model], args)
    model.to(args.device)
    model.eval()
    
    ptq_params = compute_quantized_params(model, local_rank=args.local_rank, log_file=args.log_file)
    _write('FP32 model size is {:.3f}'.format(ptq_params))
    base_acc = validate(model, loader_eval)
    _write('FP32   eval_acc: {:.2f}'.format(base_acc.avg))

    # Quantize
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(args.device).eval()

    # Initial quantization pass
    _write('Performing initial quantization ...')
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad(): _ = q_model(calib_data)

    # Scale reparameterization (unchanged)
    _write('Performing scale reparameterization ...')
    with torch.no_grad():
        module_dict = {}
        q_model_slice = q_model.layers if 'swin' in args.model else q_model.blocks
        for name, module in q_model_slice.named_modules():
            module_dict[name] = module
            idx = name.rfind('.'); idx = 0 if idx == -1 else idx
            father_name = name[:idx]
            father_module = module_dict.get(father_name, None)
            if father_module is None: continue

            is_norm1 = 'norm1' in name
            is_norm2 = 'norm2' in name
            is_plain_norm = ('norm' in name) and hasattr(father_module, 'reduction')

            if not (is_norm1 or is_norm2 or is_plain_norm): continue

            if not hasattr(module, 'weight') or module.weight is None: continue
            if not hasattr(module, 'bias')   or module.bias   is None: continue

            if is_norm1:
                next_module = father_module.attn.qkv
            elif is_norm2:
                next_module = father_module.mlp.fc1
            else:
                next_module = father_module.reduction

            if not (hasattr(next_module, 'weight') and hasattr(next_module, 'input_quantizer')): continue

            act_delta = next_module.input_quantizer.delta.reshape(-1)
            act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
            target_delta = torch.mean(act_delta)
            target_zero_point = torch.mean(act_zero_point)
            target_min = -target_zero_point * target_delta
            r = act_delta / target_delta
            act_min = -act_zero_point * act_delta
            b = act_min / r - target_min

            module.weight.data = module.weight.data / r
            module.bias.data   = module.bias.data   / r - b

            next_module.weight.data = next_module.weight.data * r
            if next_module.bias is not None:
                next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1, 1)).reshape(-1)
            else:
                next_module.bias = nn.Parameter(torch.mm(next_module.weight.data, b.reshape(-1, 1)).reshape(-1))

            next_module.input_quantizer.channel_wise = False
            next_module.input_quantizer.delta = target_delta
            next_module.input_quantizer.zero_point = target_zero_point
            next_module.weight_quantizer.inited = False

    # Re-calibration
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad(): _ = q_model(calib_data)

    # Baseline size/acc
    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('RepQ-ViT model size is {:.3f}'.format(ptq_params))
    top1_acc_eval = validate(q_model, loader_eval)
    _write('RepQ-ViT   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    # ====== YOUR ADAPTERS instead of their linear CompensationBlock ======
    # Teacher (float)
    teacher_name = model_zoo[args.model]
    teacher = timm.create_model(teacher_name, pretrained=True).to(args.device).eval()

    # Wrap with adapters (freeze backbone)
    student = add_vit_compensation_adapters(
        q_model,
        blocks=args.blocks,
        attn_use_pw=True, attn_r=args.attn_r,
        mlp_kind=args.mlp_kind, mlp_r=args.mlp_r,
        freeze_backbone=True
    )
    # Ensure quant stays enabled in backbone
    set_quant_state(student.backbone, input_quant=True, weight_quant=True)
    ptq_params = compute_quantized_params(student, local_rank=args.local_rank, log_file=args.log_file)
    _write('RepQ-ViT + QLA model size is {:.3f}'.format(ptq_params))
    # Train ONLY adapters
    trainable_params = list(student.adapters_parameters())
    assert len(trainable_params) > 0, "No adapter params found for training."
    _write(len(trainable_params))
    student, best_acc = train_adapters(
        student=student,
        teacher=teacher,
        train_loader=loader_train,
        val_loader=loader_eval,
        trainable_params=trainable_params,
        device=args.device,
        epochs=args.epochs,
        save_prefix=f"{args.model}_w{args.w_bits}a{args.a_bits}_adapters",
        num_images=args.num_images
    )

    # Final evaluation
    top1_after = validate(student, loader_eval)
    _write('RepQ-ViT + QLA   eval_acc: {:.2f}'.format(top1_after.avg))

if __name__ == '__main__':
    main()
