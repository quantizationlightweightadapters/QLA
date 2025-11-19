#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet PTQ (percentile) + ResNetAdaptersParallelIn

Examples:
  python imagenet_ptq_with_adapters.py --model resnet50 --data_dir /data/imagenet --w_bits 6 --a_bits 6 --mode adapters
  python imagenet_ptq_with_adapters.py --mode baseline
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import os.path
import random
import socket
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.utils.data
from timm.data.dataset import ImageDataset
from timm.utils import accuracy
from torch.utils.data import Dataset
from tqdm import tqdm

# from your repo:
from quant import *  # quant_model_resnet, QuantConv2d/QuantLinear/QuantMatMul, set_quant_state
from utils import *
from utils.resnet import resnet101, resnet50, resnet18
from utils.utils import write, create_transform, create_loader, AverageMeter, \
    broadcast_tensor_from_main_process, compute_quantized_params

HOST_NAME = socket.getfqdn(socket.gethostname())
torch.backends.cudnn.benchmark = True

model_path = {
    'resnet18': 'pretrained_weights/resnet18_imagenet.pth.tar',
    'resnet50': 'pretrained_weights/resnet50_imagenet.pth.tar',
    'resnet101': 'pretrained_weights/resnet101-63fe2227.pth'
}

# -----------------------------
# Utilities
# -----------------------------
def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# -----------------------------
# adapters (parallel-in at each stage)
# -----------------------------
class ResNetAdaptersParallelIn(nn.Module):
    """
    Parallel adapters that take the SAME input as each stage.
    y_i = Layer_i(x_i);  z_i = Adp_i(x_i);  out_i = y_i + z_i

    - backbone: their percentile quantized ResNet (q_model)
      (must expose forward_before_blocks, layer1..4, avgpool, fc)
    - zero-init convs so baseline stays unchanged at start
    - keeps their return contract: forward(x) -> (t_out, logits)
    """
    def __init__(self, backbone: nn.Module, factor: int = 64, k: int = 1):
        super().__init__()
        self.backbone = backbone
        self.factor = factor
        self.k = k
        self._freeze_backbone()

        # Probe shapes using backbone pipeline
        with torch.no_grad():
            self.backbone.eval()
            dev = next(self.backbone.parameters()).device
            dummy = torch.zeros(1, 3, 224, 224, device=dev)
            x1_in = self.backbone.forward_before_blocks(dummy)     # input to layer1
            o1 = self.backbone.layer1(x1_in)
            x2_in = o1; o2 = self.backbone.layer2(x2_in)
            x3_in = o2; o3 = self.backbone.layer3(x3_in)
            x4_in = o3; o4 = self.backbone.layer4(x4_in)

        def stride_from(in_t, out_t):
            sh = max(1, in_t.shape[-2] // out_t.shape[-2])
            sw = max(1, in_t.shape[-1] // out_t.shape[-1])
            return (sh, sw)

        groups1 = max(1, x1_in.shape[1] // factor)
        groups2 = max(1, x2_in.shape[1] // factor)
        groups3 = max(1, x3_in.shape[1] // factor)
        groups4 = max(1, x4_in.shape[1] // factor)

        self.ad1 = nn.Conv2d(x1_in.shape[1], o1.shape[1], kernel_size=k,
                             stride=stride_from(x1_in, o1), padding=k//2, bias=True, groups=groups1)
        self.ad2 = nn.Conv2d(x2_in.shape[1], o2.shape[1], kernel_size=k,
                             stride=stride_from(x2_in, o2), padding=k//2, bias=True, groups=groups2)
        self.ad3 = nn.Conv2d(x3_in.shape[1], o3.shape[1], kernel_size=k,
                             stride=stride_from(x3_in, o3), padding=k//2, bias=True, groups=groups3)
        self.ad4 = nn.Conv2d(x4_in.shape[1], o4.shape[1], kernel_size=k,
                             stride=stride_from(x4_in, o4), padding=k//2, bias=True, groups=groups4)

        for m in [self.ad1, self.ad2, self.ad3, self.ad4]:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward_with_features(self, x):
        # stem output feeding layer1 per their model
        t_out = self.backbone.forward_before_blocks(x)

        # Stage 1
        x1_in = t_out
        y1 = self.backbone.layer1(x1_in)
        z1 = self.ad1(x1_in)
        o1 = y1 + z1

        # Stage 2
        x2_in = o1
        y2 = self.backbone.layer2(x2_in)
        z2 = self.ad2(x2_in)
        o2 = y2 + z2

        # Stage 3
        x3_in = o2
        y3 = self.backbone.layer3(x3_in)
        z3 = self.ad3(x3_in)
        o3 = y3 + z3

        # Stage 4
        x4_in = o3
        y4 = self.backbone.layer4(x4_in)
        z4 = self.ad4(x4_in)
        o4 = y4 + z4

        # Head
        out = self.backbone.avgpool(o4)
        out = torch.flatten(out, 1)
        logits = self.backbone.fc(out)
        return logits, (o1, o2, o3, o4), t_out

    def forward(self, x):
        logits, _, t_out = self.forward_with_features(x)
        return t_out, logits

# -----------------------------
# Validation (matches their (t_out, logits) signature)
# -----------------------------
def validate(model, loader):
    top1_m = AverageMeter()
    model.eval()
    with torch.no_grad():
        for _, (inp, target) in enumerate(loader):
            inp = inp.cuda()
            target = target.cuda()
            _, output = model(inp)  # expect (t_out, logits)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            top1_m.update(acc1.item(), output.size(0))
        top1_m.synchronize()
    _write('Test  Smples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m))
    return top1_m

# -----------------------------
# KD training for adapters
# -----------------------------
def kd_loss(student_logits, teacher_logits, T=2.5):
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t     = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T*T)

def train_adapters(adapted_model: ResNetAdaptersParallelIn, fp32_teacher, loader_train, loader_val,
                   epochs=10, lr=3e-4, weight_decay=1e-4,
                   kd_weight=0.3, ce_weight=0.7, temperature=2.5, device='cuda', num_images=40000):
    ce = nn.CrossEntropyLoss()
    teacher = copy.deepcopy(fp32_teacher).eval().to(device)
    for p in teacher.parameters(): p.requires_grad = False

    params = [p for n, p in adapted_model.named_parameters()
              if any(n.startswith(s) for s in ['ad1','ad2','ad3','ad4'])]
    for p in adapted_model.backbone.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = -1.0
    for epoch in range(1, epochs+1):
        adapted_model.train()
        running = 0.0
        step = 0
        for x, y in loader_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                t_out = teacher(x)
                if isinstance(t_out, (tuple, list)):
                    t_logits = t_out[-1]
                else:
                    t_logits = t_out

            opt.zero_grad(set_to_none=True)

            s_out = adapted_model(x)
            if isinstance(s_out, (tuple, list)):
                s_logits = s_out[-1]
            else:
                s_logits = s_out

            loss = ce_weight * ce(s_logits, y) + kd_weight * kd_loss(s_logits, t_logits, T=temperature)
            loss.backward()
            opt.step()
            running += float(loss)
            
            step += 1
            if step > num_images // loader_train.batch_size:
                _write(f'Breaking from training loop after {str(step)} steps')
                break

        sched.step()
        train_loss = running / max(1, len(loader_train))
        _write(f"[Adapters] Epoch {epoch:02d} train_loss={train_loss:.6f}")
        top1 = validate(adapted_model, loader_val).avg
        if top1 > best:
            best = top1
            torch.save(adapted_model.state_dict(), os.path.join(args.log_dir, "percentile_plus_adapters_best.pth"))
            _write(f"[Adapters] ✓ Saved best (top1={best:.3f})")
    _write(f"[Adapters] Final best top1={best:.3f}")
    return adapted_model

# -----------------------------
# Argparse & globals
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet50", choices=['resnet18', 'resnet50', 'resnet101'])
parser.add_argument('--data_dir', default='/opt/Dataset/ImageNet', type=str)

parser.add_argument('--w_bits', default=4, type=int)  # percentile PTQ
parser.add_argument('--a_bits', default=4, type=int)

parser.add_argument('--kernel_size', default=1, type=int, help='adapter conv kernel size (1 or 3)')
parser.add_argument('--factor', default=64, type=int, help='adapter grouping: groups = cin // factor')

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--num_images", default=40000, type=int)

# mode: baseline or adapters
parser.add_argument("--adapter_epochs", type=int, default=1)
parser.add_argument("--adapter_lr", type=float, default=3e-4)
parser.add_argument("--adapter_wd", type=float, default=1e-4)
parser.add_argument("--adapter_kd", type=float, default=0.2)
parser.add_argument("--adapter_ce", type=float, default=0.7)
parser.add_argument("--adapter_T", type=float, default=1.5)

args = parser.parse_args()

train_aug = 'large_scale_train'
test_aug = 'large_scale_test'
args.drop_path = 0.0
args.num_classes = 1000

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
crop_pct = 0.875

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

args.log_dir = os.path.join('checkpoint', args.model, 'PTQ_Adapters',
                            f'bs_{args.batch_size}_w_{args.w_bits}_a_{args.a_bits}_k_{args.kernel_size}_factor_{args.factor}_seed_{args.seed}')
args.log_file = os.path.join(args.log_dir, 'log.txt')

if args.local_rank == 0:
    os.makedirs(args.log_dir, exist_ok=True)
    if os.path.isfile(args.log_file):
        os.remove(args.log_file)
else:
    time.sleep(1)

_write = partial(write, log_file=args.log_file)

if args.distributed:
    _write(f'Training in distributed mode. Process {args.rank}, total {args.world_size}.')
else:
    _write('Training with a single process on 1 GPU.')
assert args.rank >= 0

# -----------------------------
# Main
# -----------------------------
def main():
    if args.local_rank == 0:
        _write(str(args))
    seed(args.seed)

    _write(f'dataset mean : {mean} & std : {std}')
    dataset_train = ImageDataset(root=os.path.join(args.data_dir, 'train'),
                                 transform=create_transform(train_aug, mean, std, crop_pct))
    dataset_eval  = ImageDataset(root=os.path.join(args.data_dir, 'val'),
                                 transform=create_transform(test_aug, mean, std, crop_pct))
    _write(f'len of train_set : {len(dataset_train)}    train_transform : {dataset_train.transform}')
    _write(f'len of eval_set  : {len(dataset_eval)}    eval_transform  : {dataset_eval.transform}')

    loader_train = create_loader(dataset_train, batch_size=args.batch_size, is_training=True, re_prob=0.0,
                                 mean=mean, std=std, num_workers=args.num_workers, distributed=args.distributed,
                                 log_file=args.log_file, drop_last=True, local_rank=args.local_rank,
                                 persistent_workers=False)

    loader_eval = create_loader(dataset_eval, batch_size=args.batch_size, is_training=False, re_prob=0.0,
                                mean=mean, std=std, num_workers=args.num_workers, distributed=args.distributed,
                                log_file=args.log_file, drop_last=False, local_rank=args.local_rank,
                                persistent_workers=False)

    # Calibration batch for percentile
    for data, _ in loader_train:
        calib_data = data.to(args.device)
        break
    broadcast_tensor_from_main_process(calib_data, args)
    _write(f'local_rank : {args.local_rank} calib_data shape : {calib_data.size()} value : {calib_data[0,0,0,:5]}')

    # Float (teacher) model
    _write('Building FP32 model ...')
    if args.model == 'resnet18':
        model = resnet18(num_classes=args.num_classes, pretrained=False)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=args.num_classes, pretrained=False)
    elif args.model == 'resnet101':
        model = resnet101(num_classes=args.num_classes, pretrained=False)
    else:
        raise NotImplementedError

    checkpoint = torch.load(model_path[args.model], map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(args.device).eval()

    fp32_model = copy.deepcopy(model).to(args.device).eval()

    # Percentile PTQ model (their path)
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model_resnet(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(args.device).eval()

    _write('Performing initial quantization (percentile calibration) ...')
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model(calib_data)

    # Sizes
    fp32_params = compute_quantized_params(fp32_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('FP32 model size is {:.3f}'.format(fp32_params))
    base_acc = validate(fp32_model, loader_eval)
    _write('FP32   eval_acc: {:.2f}'.format(base_acc.avg))
    
    ptq_params  = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('Percentile model size is {:.3f}'.format(ptq_params))
    base_acc = validate(q_model, loader_eval)
    _write('Percentile   eval_acc: {:.2f}'.format(base_acc.avg))

    _write('Wrapping percentile model with ResNetAdaptersParallelIn...')
    adapted = ResNetAdaptersParallelIn(q_model, factor=args.factor, k=args.kernel_size).to(args.device)
    ptq_params  = compute_quantized_params(adapted, local_rank=args.local_rank, log_file=args.log_file)
    _write('Percentile model + Addapters size is {:.3f}'.format(ptq_params))

    # sanity: zero-init adapters keep baseline
    top1 = validate(adapted, loader_eval)
    _write('Percentile + Adapters (zero-init) eval_acc: {:.2f}'.format(top1.avg))

    # train adapters only
    adapted = train_adapters(
        adapted, fp32_model, loader_train, loader_eval,
        epochs=args.adapter_epochs, lr=args.adapter_lr, weight_decay=args.adapter_wd,
        kd_weight=args.adapter_kd, ce_weight=args.adapter_ce, temperature=args.adapter_T,
        device=args.device,
        num_images=args.num_images
    )

if __name__ == '__main__':
    main()
