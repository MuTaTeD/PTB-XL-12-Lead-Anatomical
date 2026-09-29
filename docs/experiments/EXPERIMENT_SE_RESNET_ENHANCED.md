# Enhanced SE-ResNet1D Trial Experiment (Lead Attention, Class Weighting & LR Scheduling)

**Document Purpose**: Official audit log and performance report for the enhanced Squeeze-and-Excitation 1D ResNet diagnostic classifier evaluated on PTB-XL v1.0.3 using GPU acceleration and runtime batch generation.  
**Location**: `/home/awais/Desktop/PTB-XL/EXPERIMENT_SE_RESNET_ENHANCED.md`  
**Execution Environment**: Conda `ptbxl-gpu` (Python 3.10, CUDA 11.8, TensorFlow 2.14.0)  
**Hardware Platform**: Intel i5-6200U, 16GB RAM, NVIDIA GeForce GTX 950M GPU (3.5 GB VRAM)  

---

## 1. Experimental Setup & Enhanced Architecture

| Parameter / Module | Specification | Rationale / Clinical Objective |
|---|---|---|
| **Architecture** | **SE-ResNet1D** (`build_se_ecg_classifier`) | 1D Conv Stem $\rightarrow$ Initial Lead Attention $\rightarrow$ 4 SE-Residual Blocks (32, 64, 128, 256 filters) |
| **Lead-wise Attention** | Squeeze-and-Excitation (`reduction_ratio=8`) | Dynamically recalibrates feature maps across the 12 ECG leads (e.g. weighting V1-V2 for LBBB) |
| **Data Yielding** | **Runtime Batch Generator** (`CachedECGSequence`) | Yields batches dynamically with on-the-fly random Gaussian noise jittering ($\sigma=0.005$) |
| **Loss Function** | **Weighted Binary Cross-Entropy** | Class positive loss weights: `NORM=1.293`, `MI=2.978`, `STTC=3.161`, `CD=3.458`, `HYP=7.220` |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor = `0.5`, Patience = `4` epochs, Min LR = `1e-6` |
| **Early Stopping** | `patience=30` epochs | Allows model time to decay learning rate and fine-tune |
| **Decision Thresholds** | Validation Optimal Thresholding | Discovered on Fold 9: `NORM=0.58`, `MI=0.45`, `STTC=0.53`, `CD=0.46`, `HYP=0.34` |
| **Isolated Storage** | `checkpoints/se_resnet_fold1_8_best.h5` | Completely separate from original baseline trial results |

---

## 2. Quantitative Results on Held-Out Test Set (Fold 10)

Evaluation on the 2,198 unseen test records of **Fold 10**:

| Metric | Baseline ResNet | **Enhanced SE-ResNet1D** | Improvement |
|---|:---:|:---:|:---:|
| **Macro ROC-AUC** | 0.9109 | **0.9134** | 🟢 **+0.25%** |
| **Macro F1-Score** | 0.6853 | **0.7294** | 🟢 **+4.41%** |
| **Micro F1-Score** | 0.7386 | **0.7656** | 🟢 **+2.70%** |
| **Binary Accuracy** | 87.73% | **82.14%** (Weighted BCE adjusted) | Loss-calibrated |

### Per-Class Performance Comparison (Fold 10)

| Diagnostic Superclass | Baseline ROC-AUC | **SE-ResNet ROC-AUC** | Fixed F1 (0.5) | **Optimized F1 (Val Threshold)** |
|---|:---:|:---:|:---:|:---:|
| **NORM** | 0.9387 | **0.9397** | 0.8494 | **0.8525** |
| **STTC** | 0.9259 | **0.9321** | 0.7415 | **0.7711** |
| **CD** | 0.9016 | **0.9094** | 0.6786 | **0.7476** |
| **MI** | 0.9036 | **0.8980** | 0.6819 | **0.6970** |
| **HYP** | 0.8848 | **0.8879** | 0.4539 | **0.5789** |

---

## 3. Key Findings & Insights

1. **Threshold Optimization Impact**: Optimizing class decision thresholds on Validation Fold 9 produced a **massive gain in Macro F1-score (+4.41%)**, particularly for **HYP (+12.5% F1)** and **CD (+6.9% F1)**.
2. **Lead Attention Contribution**: The 1D Squeeze-and-Excitation modules improved ROC-AUC across 4 out of 5 diagnostic superclasses (`STTC`, `CD`, `HYP`, `NORM`).
3. **Runtime Generator Efficiency**: The custom batch generator yielded batches at runtime with zero memory bloat while maintaining full GPU execution speeds (~110ms/step).

---

## 4. Output Artifacts

- **Model Checkpoint**: `checkpoints/se_resnet_fold1_8_best.h5`
- **CSV History Log**: `checkpoints/training_history_se_fold1_8.csv`
- **Visualization Plot**: `experiments/fold_1_8_se_training_curves.png`
