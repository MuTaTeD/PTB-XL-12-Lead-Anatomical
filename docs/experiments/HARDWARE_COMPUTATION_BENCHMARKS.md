# Hardware Computation & Training Benchmark Guide: 1D DDPM Counterfactual U-Net

**Document Purpose**: Performance, memory, and runtime estimation guide across available GPU workstation and desktop configurations for training the 1D Conditional DDPM U-Net model on PTB-XL (17,441 training signals per epoch).

---

## 1. Comprehensive Hardware Comparison Table

| Machine Option | GPU & System Hardware Specifications | Est. Time per Epoch | **Single Fold (15 Epochs)** | **Full 10-Fold CV (150 Epochs)** | Max Batch Size | FP16 Precision | Speedup vs Base (GTX 950M) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 👑 **Workstation (A4000)** | **NVIDIA RTX A4000 (16GB VRAM)**<br>Xeon CPU, 32/64GB RAM | ⚡ **~15 – 25 sec** | 🚀 **~4 – 6 Minutes** | 🌟 **~45 – 60 Minutes** | **512 / 1024** | 🟢 Native Tensor FP16 | 🚀 **~18x – 25x Faster** |
| 🖥️ **Desktop 2 (GTX 1080)** | **NVIDIA GTX 1080 (8GB VRAM)**<br>i5 6th/7th Gen, 16GB RAM | ~1.0 – 1.3 min | 🟢 **~15 – 20 Minutes** | **~2.5 – 3.0 Hours** | **256** | FP32 / Standard | 🚀 **~8x Faster** |
| 💻 **Laptop (RTX 3050)** | **NVIDIA RTX 3050 (4GB VRAM)**<br>i7 11th Gen, 32GB RAM | ~1.2 – 1.5 min | 🟢 **~18 – 22 Minutes** | **~3.0 – 3.5 Hours** | **128** | 🟢 Ampere FP16 | 🚀 **~7x Faster** |
| 🖥️ **Desktop 1 (GTX 1650)** | **NVIDIA GTX 1650 (4GB VRAM)**<br>i5 4th/5th Gen, 8–16GB RAM | ~2.5 – 3.0 min | **~40 – 45 Minutes** | **~6.5 – 7.5 Hours** | **128** | FP32 / Standard | 🚀 **~3.5x Faster** |
| 💻 **Current Machine** | GTX 950M (2GB VRAM / CPU) | ~10.3 min | **~2.5 Hours** | ~25.5 Hours | 64 / 128 | FP32 / Standard | 1x (Baseline) |

---

## 2. Detailed Breakdown & Recommendation

### 🏆 1st Choice: Workstation with NVIDIA RTX A4000 (16GB VRAM)
- **Why**: 16GB VRAM and 6,144 Ampere CUDA cores allow enabling TensorFlow FP16 Mixed Precision (`tf.keras.mixed_precision.set_global_policy('mixed_float16')`) with **Batch Size 512**.
- **Runtime**: Completes single-fold 15-epoch training in **~5 minutes** and entire 10-Fold CV in **under 1 hour**!
- **Best Use Case**: Running full 10-Fold Cross-Validation, zero-shot external dataset validation (CPSC2018), and full $t^*$ noise trajectory guidance ablation studies in one afternoon.

### 🥈 2nd Choice: Desktop 2 with NVIDIA GTX 1080 (8GB VRAM)
- **Why**: 8GB VRAM provides double the memory capacity of 4GB GPUs, avoiding Out-Of-Memory (OOM) risks and allowing **Batch Size 256**.
- **Runtime**: Single fold completes in **~15–20 minutes**; full 10-Fold CV in **~2.5 hours** overnight.

### 🥉 3rd Choice: Laptop with NVIDIA RTX 3050 (4GB VRAM, i7 11th Gen)
- **Why**: High single-thread CPU performance from the 11th Gen i7 CPU ensures zero data-loading bottleneck.
- **Runtime**: Single fold completes in **~20 minutes**.

---

## 3. How to Enable FP16 Mixed Precision Acceleration

For GPUs with Tensor Cores (RTX A4000 & RTX 3050), add the following lines at the top of your training script for up to **2x additional speedup**:

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable FP16 Mixed Precision for Tensor Core GPUs
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision policy set to float16")
```
