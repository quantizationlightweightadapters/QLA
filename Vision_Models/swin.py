#!/usr/bin/env python3
import argparse, copy, math, os, random, socket
from functools import partial
from typing import Optional, Sequence, Literal, Dict, Tuple, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import timm
from timm.data.dataset import ImageDataset
from timm.utils import accuracy as timm_accuracy

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.scheduler_factory import CosineLRScheduler

# --- your stack ---
from quant import *
from utils import *
from utils.utils import write, create_transform, create_loader, AverageMeter, broadcast_tensor_from_main_process, gather_tensor_from_multi_processes, compute_quantized_params

HOST_NAME = socket.getfqdn(socket.gethostname())
torch.backends.cudnn.benchmark = True
LINEAR_COMPENSATION_SAMPLES = 512  # only used for their path; harmless here

# ============================================================
# Utilities carried over
# ============================================================
def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def enable_quant(submodel):
    for _, m in submodel.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(True, True)

def disable_quant(submodel):
    for _, m in submodel.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(False, False)

# ============================================================
# ========  YOUR ADAPTERS for SWIN (DW for W-MSA, LoRA for MLP)
# ============================================================
class Gate(nn.Module):
    def __init__(self, init=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x): return self.alpha * x

class DepthwiseSeparableWindowAdapter(nn.Module):
    """
    Spatial DW(+PW) adapter computed on the full HxW grid and added to the
    attention residual. Expects x_tok: (B, L, C) with L = H*W (pre-attn input).
    """
    def __init__(self, dim: int, use_pw: bool = True, r: int = 32, act: str = "gelu", gate_init: float = 0.1):
        super().__init__()
        self.dim = dim; self.use_pw = use_pw
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)
        if use_pw:
            self.pw1 = nn.Conv2d(dim, r, kernel_size=1, bias=True)
            self.pw2 = nn.Conv2d(r, dim, kernel_size=1, bias=True)
            nn.init.kaiming_uniform_(self.pw1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.pw2.weight)
            nn.init.zeros_(self.pw1.bias); nn.init.zeros_(self.pw2.bias)
        else:
            self.pw1 = self.pw2 = None
        self.act = nn.GELU() if act == "gelu" else (nn.SiLU() if act == "silu" else nn.Identity())
        self.gate = Gate(gate_init)

    def forward(self, x_tok: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x_tok.shape
        assert L == H * W, f"Adapter expects L=H*W; got L={L}, H*W={H*W}"
        x_grid = x_tok.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        y = self.dw(x_grid)
        y = self.act(y)
        if self.use_pw:
            y = self.pw2(self.act(self.pw1(y)))
        y_tok = y.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return self.gate(y_tok)

class MLPTokenAdapter(nn.Module):
    """
    Token-wise LoRA-style adapter for MLP branch.
    kind: "affine" | "lowrank" | "bottleneck"
    """
    def __init__(self, dim: int, kind: Literal["affine","lowrank","bottleneck"]="lowrank",
                 r: int = 32, act: str = "gelu", gate_init: float = 0.1):
        super().__init__()
        self.dim = dim; self.kind = kind
        if kind == "affine":
            self.scale = nn.Parameter(torch.ones(dim))
            self.bias  = nn.Parameter(torch.zeros(dim))
            self.act = nn.Identity()
            self.low1 = self.low2 = None; self.mid = None
        else:
            self.act = nn.GELU() if act == "gelu" else (nn.SiLU() if act == "silu" else nn.Identity())
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "affine":
            y = x * self.scale + self.bias
        elif self.kind == "lowrank":
            y = self.low2(self.low1(x))
        else:
            y = self.low2(self.act(self.mid(self.act(self.low1(x)))))
        return self.gate(y)

# --- helpers copied from Swin ---
def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WrappedSwinBlock(nn.Module):
    """
    Inserts: DW(+PW) adapter at attention residual, LoRA-like adapter at MLP residual.
    Faithfully replays the block's forward (pre-LN, shift/windows/mask).
    """
    def __init__(self, block, dim: int,
                 attn_use_pw: bool=True, attn_r: int=32, attn_act: str="gelu", attn_gate: float=0.1,
                 mlp_kind: str="lowrank", mlp_r: int=32, mlp_act: str="gelu", mlp_gate: float=0.1):
        super().__init__()
        self.block = block
        self.dim = dim
        self.attn_adapter = DepthwiseSeparableWindowAdapter(dim, use_pw=attn_use_pw, r=attn_r, act=attn_act, gate_init=attn_gate)
        self.mlp_adapter  = MLPTokenAdapter(dim, kind=mlp_kind, r=mlp_r, act=mlp_act, gate_init=mlp_gate)
        self.has_dp = hasattr(block, 'drop_path') and isinstance(block.drop_path, nn.Module)
        self.input_resolution = tuple(block.input_resolution) if hasattr(block, "input_resolution") else None
        self.shift_size = getattr(block, "shift_size", 0)
        self.window_size = getattr(block, "window_size", 7)
        self.attn_mask = getattr(block, "attn_mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and C == self.dim, f"Swin adapter expects (B,{H*W},{self.dim}); got {x.shape}"

        # ---- Attention branch ----
        u = self.block.norm1(x)                  # (B, L, C)
        uc = u.view(B, H, W, C)
        if self.shift_size > 0:
            uc = torch.roll(uc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        uw = window_partition(uc, self.window_size)                          # (nW*B, ws, ws, C)
        uw = uw.view(-1, self.window_size * self.window_size, C)            # (nW*B, ws*ws, C)
        attn_out = self.block.attn(uw, mask=self.attn_mask)                 # (nW*B, ws*ws, C)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        uc_out = window_reverse(attn_out, self.window_size, H, W)           # (B, H, W, C)
        if self.shift_size > 0:
            uc_out = torch.roll(uc_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        a = uc_out.view(B, H * W, C)                                        # token layout

        a_hat = self.attn_adapter(u, H, W)                                   # DW(+PW) on pre-attn input
        a_sum = a + a_hat
        x = x + (self.block.drop_path(a_sum) if self.has_dp else a_sum)

        # ---- MLP branch ----
        v = self.block.norm2(x)
        m = self.block.mlp(v)
        m_hat = self.mlp_adapter(v)
        m_sum = m + m_hat
        x = x + (self.block.drop_path(m_sum) if self.has_dp else m_sum)
        return x

class SwinQuantWithAdapters(nn.Module):
    """
    Wrap a (quantized) timm Swin model; insert adapters into chosen (stage, block) pairs or all.
    """
    def __init__(self, quant_swin: nn.Module,
                 blocks_to_adapt=None,
                 attn_use_pw=True, attn_r=32, attn_act="gelu", attn_gate=0.1,
                 mlp_kind="lowrank", mlp_r=32, mlp_act="gelu", mlp_gate=0.1,
                 freeze_backbone=True):
        super().__init__()
        self.backbone = quant_swin
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        stages = getattr(self.backbone, "layers", None)
        if stages is None:
            raise AttributeError("Expected Swin model with attribute 'layers'")

        want = "all" if (blocks_to_adapt is None or blocks_to_adapt == "all") else set(tuple(x) for x in blocks_to_adapt)

        for s_idx, stage in enumerate(stages):
            blk_list = getattr(stage, "blocks", None)
            if blk_list is None: continue
            new_blks = []
            for b_idx, blk in enumerate(blk_list):
                dim = blk.norm1.normalized_shape[0]
                if want == "all" or (s_idx, b_idx) in want:
                    new_blks.append(
                        WrappedSwinBlock(
                            blk, dim,
                            attn_use_pw=attn_use_pw, attn_r=attn_r, attn_act=attn_act, attn_gate=attn_gate,
                            mlp_kind=mlp_kind, mlp_r=mlp_r, mlp_act=mlp_act, mlp_gate=mlp_gate
                        )
                    )
                else:
                    new_blks.append(blk)
            stage.blocks = nn.ModuleList(new_blks)

    def forward(self, x):
        return self.backbone(x)

    def adapters_parameters(self):
        for m in self.modules():
            if isinstance(m, (DepthwiseSeparableWindowAdapter, MLPTokenAdapter, Gate)):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p

def add_swin_compensation_adapters(sim_model: nn.Module,
                                   blocks=None,          # e.g. [(0,2),(1,0)] or "all"
                                   attn_use_pw=True, attn_r=32,
                                   mlp_kind="lowrank", mlp_r=32,
                                   freeze_backbone=True) -> SwinQuantWithAdapters:
    return SwinQuantWithAdapters(
        quant_swin=sim_model,
        blocks_to_adapt=blocks if blocks is not None else "all",
        attn_use_pw=attn_use_pw, attn_r=attn_r,
        mlp_kind=mlp_kind, mlp_r=mlp_r,
        freeze_backbone=freeze_backbone
    )

# ============================================================
# ========  Training bits (KD + CE), robust validate
# ============================================================
class DistillLoss(nn.Module):
    def __init__(self, tau=2.0, alpha_kd=0.5, alpha_fm=0.0):
        super().__init__()
        self.tau = tau
        self.alpha_kd = alpha_kd
        self.alpha_fm = alpha_fm
        self.ce = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction="batchmean")
    def forward(self, logits_s, logits_t, labels, feats_s=None, feats_t=None):
        loss_ce = self.ce(logits_s, labels)
        t = self.tau
        log_p_s = F.log_softmax(logits_s / t, dim=1)
        p_t     = F.softmax(logits_t / t, dim=1)
        loss_kd = (t * t) * self.kld(log_p_s, p_t)
        loss_fm = 0.0
        if self.alpha_fm > 0 and feats_s and feats_t:
            for k in feats_s.keys():
                if k in feats_t:
                    loss_fm = loss_fm + F.mse_loss(feats_s[k], feats_t[k])
        return loss_ce + self.alpha_kd * loss_kd + self.alpha_fm * (loss_fm if isinstance(loss_fm, torch.Tensor) else 0.0)

def set_trainable_adapters_only(model):
    for p in model.parameters(): p.requires_grad_(False)
    for n, m in model.named_modules():
        if any(k in n for k in ("attn_adapter", "mlp_adapter", "gate")):
            for p in m.parameters(): p.requires_grad_(True)

def evaluate_top1(model, val_loader, device="cuda", amp=True):
    model.eval()
    correct, total = 0, 0
    use_amp = amp and (device == "cuda" or getattr(device, "type", "") == "cuda")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple):  # (feat, logits) safety
                logits = out[-1]
            else:
                logits = out
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def validate(model, loader):
    top1_m = AverageMeter(); model.eval()
    with torch.no_grad():
        for batch_idx, (inp, target) in enumerate(loader):
            inp = inp.cuda(); target = target.cuda()
            out = model(inp)
            logits = out[-1] if isinstance(out, tuple) else out
            acc1, _ = timm_accuracy(logits, target, topk=(1, 5))
            top1_m.update(acc1.item(), logits.size(0))
        top1_m.synchronize()
    _write('Test  Smples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m))
    return top1_m

def train_adapters(
    student,
    teacher,
    train_loader,
    val_loader=None,
    device="cuda",
    epochs=1,
    lr=5e-4,
    weight_decay=0.0,
    amp=True,
    tau=2.0,
    alpha_kd=0.5,
    alpha_fm=0.0,       # default off for simplicity
    grad_accum=1,
    max_norm=None,
    save_prefix="swin_quant_adapters",
    num_images=40000
):
    student.to(device).train()
    teacher.to(device).eval()
    set_trainable_adapters_only(student)

    trainable_params = list(student.adapters_parameters())
    assert len(trainable_params) > 0, "No adapter params to train."

    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    use_amp = amp and (device == "cuda" or getattr(device, "type", "") == "cuda")
    scaler = GradScaler('cuda', enabled=use_amp)
    crit = DistillLoss(tau=tau, alpha_kd=alpha_kd, alpha_fm=alpha_fm)

    best_acc, best_state = 0.0, None
    step = 0
    for epoch in range(epochs):
        student.train()
        epoch_step = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast('cuda', enabled=use_amp):
                with torch.no_grad():
                    logits_t = teacher(images)
                    if isinstance(logits_t, tuple): logits_t = logits_t[-1]
                logits_s = student(images)
                if isinstance(logits_s, tuple): logits_s = logits_s[-1]
                loss = crit(logits_s, logits_t, labels) / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % grad_accum == 0:
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

    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    return student, best_acc

# ============================================================
# ========  Main script (Swin PTQ -> wrap -> train adapters)
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="swin_tiny", choices=['swin_tiny', 'swin_small'], help="model")
parser.add_argument('--data_dir', default='/opt/Dataset/ImageNet', type=str)

parser.add_argument('--w_bits', default=4, type=int, help='bit-precision of weights')
parser.add_argument('--a_bits', default=4, type=int, help='bit-precision of activation')
parser.add_argument('--blocks', nargs='+', help='Blocks to add QLA layers to', default='all')

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--local-rank", default=0, type=int)

# adapter hyperparams (tweakable from CLI)
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
args.rank = 0
if args.distributed:
    args.device = f'cuda:{args.local_rank}'
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
assert args.rank >= 0

args.log_dir = os.path.join('checkpoint', args.model, 'FAIR_SWIN_ADAPTERS',
                            f'bs_{args.batch_size}_worldsize_{args.world_size}_w_{args.w_bits}_a_{args.a_bits}_sed_{args.seed}')
args.log_file = os.path.join(args.log_dir, 'log.txt')
if args.local_rank == 0:
    os.makedirs(args.log_dir, exist_ok=True)
    if os.path.isfile(args.log_file):
        os.remove(args.log_file)
torch.cuda.synchronize()
_write = partial(write, log_file=args.log_file)

def main():
    if args.local_rank == 0:
        _write(vars(args))
    seed_everything(args.seed)

    if args.local_rank == 0:
        _write('dataset mean : {} & std : {}'.format(mean, std))

    # Datasets / loaders
    dataset_train = ImageDataset(root=os.path.join(args.data_dir, 'train'),
                                 transform=create_transform(train_aug, mean, std, crop_pct))
    dataset_eval  = ImageDataset(root=os.path.join(args.data_dir, 'val'),
                                 transform=create_transform(test_aug,  mean, std, crop_pct))
    if args.local_rank == 0:
        _write('len of train_set : {}    train_transform : {}'.format(len(dataset_train), dataset_train.transform))
        _write('len of eval_set  : {}    eval_transform  : {}'.format(len(dataset_eval), dataset_eval.transform))

    loader_train = create_loader(
        dataset_train, batch_size=args.batch_size, is_training=True, re_prob=0.0, mean=mean, std=std,
        num_workers=args.num_workers, distributed=args.distributed, log_file=args.log_file,
        drop_last=True, local_rank=args.local_rank, persistent_workers=False
    )
    loader_eval = create_loader(
        dataset_eval, batch_size=args.batch_size, is_training=False, re_prob=0.0, mean=mean, std=std,
        num_workers=args.num_workers, distributed=args.distributed, log_file=args.log_file,
        drop_last=False, local_rank=args.local_rank, persistent_workers=False
    )

    # Calibration batch
    for data, _ in loader_train:
        calib_data = data.to(args.device)
        break
    broadcast_tensor_from_main_process(calib_data, args)
    _write('local_rank : {} calib_data shape : {} value : {}'.format(
        args.local_rank, calib_data.size(), calib_data[0, 0, 0, :5]))

    model_zoo = {
        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
    }

    # ----- Build & Quantize (RepQ-ViT style exactly as in your script) -----
    _write('Building model ...')
    model = build_model(model_zoo[args.model], args).to(args.device).eval()

    ptq_params = compute_quantized_params(model, local_rank=args.local_rank, log_file=args.log_file)
    _write('FP32 model size is {:.3f}'.format(ptq_params))
    base_acc = validate(model, loader_eval)
    _write('FP32   eval_acc: {:.2f}'.format(base_acc.avg))

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params).to(args.device).eval()

    # Initial quantization
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
            if not (hasattr(module, 'weight') and hasattr(module, 'bias')): continue
            if module.weight is None or module.bias is None: continue

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

    # Baseline PTQ size/acc
    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('RepQ-ViT (PTQ) model size is {:.3f}'.format(ptq_params))
    base_acc = validate(q_model, loader_eval)
    _write('RepQ-ViT (PTQ)   eval_acc: {:.2f}'.format(base_acc.avg))

    # ===== Replace their CompensationBlock with YOUR adapters =====
    student = add_swin_compensation_adapters(
        q_model,
        blocks=args.blocks,                      # or a list of (stage_idx, block_idx)
        attn_use_pw=True, attn_r=args.attn_r,
        mlp_kind=args.mlp_kind, mlp_r=args.mlp_r,
        freeze_backbone=True
    )
    ptq_params = compute_quantized_params(student, local_rank=args.local_rank, log_file=args.log_file)
    _write('Swin RepQ-ViT (PTQ) model with Adapters size is {:.3f}'.format(ptq_params))
    # keep quantization active in backbone
    set_quant_state(student.backbone, input_quant=True, weight_quant=True)

    # Teacher: float Swin of same family
    teacher_name = model_zoo[args.model]
    teacher = timm.create_model(teacher_name, pretrained=True).to(args.device).eval()

    # Train adapters only (KD + CE)
    student, best_acc = train_adapters(
        student=student,
        teacher=teacher,
        train_loader=loader_train,
        val_loader=loader_eval,
        device=args.device,
        epochs=args.epochs,
        save_prefix=f"{args.model}_w{args.w_bits}a{args.a_bits}_swin_adapters",
        num_images=args.num_images
    )

    # Final eval
    top1_after = validate(student, loader_eval)
    _write('PTQ + QLA   eval_acc: {:.2f}'.format(top1_after.avg))

if __name__ == '__main__':
    main()
