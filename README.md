# QLA: Fixing Quantization with Lightweigh Adapters

Official implementation for our **CVPR 2026 submission**:  
> *"Fixing Quantization with Lightweigh Adapters"*

QLA is a fast, simple, and effective method to improve the performance of any arbitrary post training quantized model. QLA work by inserting small depthwise-separable adapters in attention/convolutional layers and low-rank adapters in feed-forward layers to correct quantization-induced errors. The adapters are optimized via a hybrid objective that combines supervised learning, knowledge distillation, and feature reconstruction, enabling accurate alignment with the full-precision model. Our approach requires only minimal tuning on a small subset of the training data and adds less than 1\% overhead for large models and around 10\% for tiny models, yet effectively recovers performance lost to low-bit quantization.

## 🧩 Architecture Overview

<p align="center">
  <img src="images/overal.png" width="40%" alt="Overview of the proposed lightweight adapter architecture: DW-separable conv on the attention branch and LoRA on the FFN branch, integrated via residual Add&Norm." />
</p>
<p align="center"><em>
Figure 1 — Overview. We attach a depthwise–separable (DW-Sep) convolution in parallel to the self-attention path to correct spatial distortions, and a LoRA block on the FFN path to correct low-rank channel errors. Both outputs are merged through the residual Add&Norm, preserving the original block I/O shape.
</em></p>

<p align="center">
  <img src="images/detailed2.png" width="60%" alt="Detailed block diagram showing where DW-separable conv and LoRA attach relative to Self-Attention, Intermediate/Output FFN, and Add&Norm." />
</p>
<p align="center"><em>
Figure 2 — Per-block wiring. DW-Sep receives the same block input as attention and is added back post-attention; LoRA augments the FFN output. Dashed arrows denote residual fusion; solid arrows denote the quantized backbone flow.
</em></p>



## 🖼️ Vision Models
This section includes image classification models adapted with our quantization and compensation framework:
- **ResNet-50 + QLA (PTQ + Depthwise-Separable Compensation)**
- **DeiT / ViT + QLA (RepQ + Adapter Compensation)**
- **Swin-Tiny + QLA (RepQ + Adapter Compensation)**

All models are implemented in PyTorch

👉 [**See details**](Vision_Models/readme.md)

---

## 💬 Language Models
Quantized NLP backbones fine-tuned for:
- **Question Answering (SQuAD v1.1)** — RoBERTa / GPT-2 with RepQ or BNB quantization
- **Semantic Similarity (MRPC)** — BERT-Base with PTQ

👉 [**See details**](Language_Models/readme.md)

---

## ⚙️ Setup
```bash
conda create -n qla python=3.8
conda activate qla
pip install -r requirements.txt
```

## Acknowledgement
This repository was built upon Quantization without Tears and RepQ-ViT. Links are not currently present as they are not allowed in CVPR submissions.
