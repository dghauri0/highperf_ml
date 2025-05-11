# 🧠 Multimodal LLM Optimization: Cross-Modality Understanding

A high-performance Visual Question Answering (VQA) system combining **ViT** for image processing and **BERT** for text analysis. This project builds efficient multimodal models through distributed training, knowledge distillation, and architecture optimizatiomn.

---

## 🚀 Highlights

- Unified embedding-decoder architecture for aligned image-text representations
- ViT + BERT for teacher model, ResNet18 + DistilBERT for compact student
- Distributed training with **4×L4 GPUs** via DDP (best performance observed)
- Knowledge distillation for compression (~60% model size reduction)
- Retained multi-modal understanding with faster inference
- Quantization explored (not deployed due to CUDA/DDP limitations)

---

## 🎯 Project Goals

- Accelerate training and improve stability using multi-GPU setups
- Reduce model complexity while maintaining accuracy
- Deploy efficient student models suitable for edge devices
- Investigate optimization tools (AMP, gradient checkpointing, quantization)

---

## ✅ Milestones

- [x] Define multimodal architecture (ViT + BERT, shared embeddings)
- [x] Prototype on Google Colab (1×L4 GPU)
- [x] Migrate to NYU HPC with DDP (4×L4 GPUs)
- [x] Optimize training (loss ~0.098, smooth convergence)
- [x] Apply knowledge distillation (ResNet18 + DistilBERT)
- [ ] Quantization Deployment ❌ *Explored, but blocked by compatibility issues*
- [ ] Cross-Attention Fusion 😞 *Deferred due to memory/model complexity*
- [x] Benchmark performance across 1-GPU, 2-GPU, and 4-GPU setups

---

## 📈 Key Results

| Setup         | Final Loss | Speedup vs. 4-GPU | Notes                      |
|---------------|------------|-------------------|----------------------------|
| 1×L4 GPU       | ~0.52      | ~0.36×            | Fastest steps/sec, poor convergence |
| 2×L4 GPU       | ~1.44      | ~0.96×            | Underperformed, syncing issues? |
| 4×L4 GPU (DDP) | **~0.098** | 1.0× (baseline)   | Best overall performance   |

---

## 🗂️ Repository

🔗 [GitHub Project](https://github.com/dghauri0/highperf_ml)

---

## 🙌 Authors

- dg4140 
- ac11274
