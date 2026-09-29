# Single-Fold Trial Experiment: 1D ResNet ECG Diagnostic Classifier

**Document Purpose**: Official audit log and performance report for the single-fold trial experiment executed on PTB-XL v1.0.3 using GPU acceleration.  
**Location**: `/home/awais/Desktop/PTB-XL/EXPERIMENT_SINGLE_FOLD_RESNET.md`  
**Execution Environment**: Conda `ptbxl-gpu` (Python 3.10, CUDA 11.8, TensorFlow 2.14.0, cuDNN 8.9)  
**Hardware Platform**: Intel i5-6200U, 16GB RAM, NVIDIA GeForce GTX 950M GPU (3.5 GB allocated VRAM)  

---

## 1. Experimental Setup & Protocol

| Hyperparameter / Setup Metric | Specification / Value | Description / Rationale |
|---|---|---|
| **Training Dataset** | **Folds 1–8** (17,418 ECG records) | Pre-computed patient-grouped `strat_fold` split |
| **Validation Dataset** | **Fold 9** (2,183 ECG records) | Used for Early Stopping and Model Checkpointing |
| **Test Dataset** | **Fold 10** (2,198 ECG records) | Held-out evaluation benchmark set |
| **Model Architecture** | 1D ResNet Classifier | Conv1D Stem $\rightarrow$ 4 Residual Blocks (32, 64, 128, 256 filters) $\rightarrow$ GAP $\rightarrow$ Dense(128) $\rightarrow$ Sigmoid(5) |
| **Max Epochs** | `100` | Early stopping monitored on `val_loss` |
| **Batch Size** | `64` | Optimized for GTX 950M VRAM capacity |
| **Optimizer** | `Adam` ($\text{learning\_rate} = 10^{-3}$) | Adaptive moment estimation |
| **Loss Function** | Binary Cross-Entropy | Multi-label 5-superclass classification |
| **Callbacks** | `ModelCheckpoint`, `EarlyStopping`, `CSVLogger` | Best weights saved to `checkpoints/resnet_fold1_8_best.h5` |
| **Resumption System** | State-aware auto-resumption | Automatically recovers weights and log state if stopped mid-way |

---

## 2. Experimental Execution & Mid-Way Recovery Validation

1. **Fast Binary Preloading**:
   - Built binary compressed dataset cache `dataset-1.0.3/ptbxl_100hz_cached.npz` (896 MB) using `ProcessPoolExecutor` multi-processing.
   - Dataset loading time reduced from ~70s to **0.25 seconds**.
2. **Mid-Way Interruption & Restoration Audit**:
   - Intentionally terminated execution at Epoch 5.
   - Upon relaunching `run_single_fold_experiment.py`, the system successfully detected existing checkpoint `checkpoints/resnet_fold1_8_best.h5` and CSV log `training_history_fold1_8.csv`.
   - Script printed: `[Checkpoint Recovery] Resuming training from Epoch 5/100` and resumed fit seamlessly.
3. **Early Stopping Trigger**:
   - Model achieved lowest validation loss at Epoch 5 (`val_loss`: `0.3033`, `val_auc`: `0.9173`).
   - Early stopping patience (`15` epochs) monitored performance until Epoch 20, whereupon training halted and best weights were automatically restored.

---

## 3. Performance Metrics on Held-Out Test Set (Fold 10)

Evaluation on the 2,198 unseen ECG records of **Fold 10**:

| Metric | Score / Value |
|---|:---:|
| **Overall Test Loss** | **0.3162** |
| **Overall Binary Accuracy** | **87.73%** |
| **Overall Macro ROC-AUC** | **0.9109** |
| **Overall Macro F1-Score** | **0.6853** |
| **Overall Micro F1-Score** | **0.7386** |

### Per-Class Diagnostic Superclass ROC-AUC (Fold 10)

| Diagnostic Superclass | Full Name | ROC-AUC Score |
|---|---|:---:|
| **NORM** | Normal ECG | **0.9387** |
| **STTC** | ST/T Changes | **0.9259** |
| **MI** | Myocardial Infarction | **0.9036** |
| **CD** | Conduction Disturbances | **0.9016** |
| **HYP** | Hypertrophy | **0.8848** |

---

## 4. Visualizations & Generated Artifacts

- **Model Checkpoint**: `checkpoints/resnet_fold1_8_best.h5`
- **Training CSV Log**: `checkpoints/training_history_fold1_8.csv`
- **Training Curves Plot**: `experiments/fold_1_8_training_curves.png`

---
*Document compiled for research audit.*
