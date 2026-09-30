# Experiment & Implementation Log: Clinical ECG Layout Standardization and Duplicate Label Resolution

**Date**: 2026-09-30  
**Authors**: Antigravity & User  
**Target Script**: `evaluate_anatomical_xai.py`  
**Output Figures**: `manuscript/figures/fig3_xai_mi.png` through `fig7_xai_norm.png`  
**Execution Environment**: Conda `ptbxl-gpu` (Python 3.10, TensorFlow 2.14.0, CUDA 11.8, cuDNN 8.9, NVIDIA GeForce GTX 950M 4GB)

---

## 1. Executive Summary

This log records the resolution of two visual and structural issues in the multi-panel Explainable AI (XAI) clinical visualization suite:
1. **Resolution of Duplicate Lead Labels in Grad-CAM++ (Section A) and Integrated Gradients (Section B)**:
   - *Previous state*: Lead names appeared twice per trace—once outside the bounding box as Y-axis tick labels, and once inside the plot area near the 1st vertical grid line (at $x = \text{start\_t} - 0.04$).
   - *Fix*: Removed hardcoded internal `ax.text(...)` and `ax_ig.text(...)` calls. Directly formatted the standard Matplotlib Y-axis ticklabels with anatomical territory colors and bold 9.5 pt font outside the clinical grid box.
2. **Clinical 2-Column ECG Standardization**:
   - Replaced vertical stacking of 12 leads into a two-column clinical standard layout:
     - **Column 1**: Standard 6 Limb Leads ($I, II, III, aVR, aVL, aVF$)
     - **Column 2**: Standard 6 Precordial Chest Leads ($V_1, V_2, V_3, V_4, V_5, V_6$)
   - Preserved authentic clinical paper proportions:
     - **Paper Speed**: 25 mm/s $\rightarrow$ 0.20 s major grid lines, 0.04 s minor grid lines.
     - **Voltage Calibration**: 10 mm/mV $\rightarrow$ 0.50 mV major grid lines, 0.10 mV minor grid lines.
     - **Calibration Pulse**: Standard 1.0 mV square calibration pulse rendered on the left margin.
   - Dual-trace rendering: Dark charcoal physiological ECG trace (`#111827`, 0.9 pt) underneath, overlaid with 1D Grad-CAM++ saliency LineCollection (`plasma` colormap, 2.0 pt).
3. **Hardware Execution Analysis (GPU vs. CPU)**:
   - Comprehensive profiling of `ptbxl` (CPU) vs. `ptbxl-gpu` (GTX 950M 4GB).

---

## 2. Duplicate Label Root Cause and Resolution

### Root Cause
In `evaluate_anatomical_xai.py`, the plotting function configured the Y-axis ticklabels outside the plot box:
```python
ax.set_yticks(offsets)
ax.set_yticklabels([LEAD_NAMES[i] for i in indices], fontsize=8.5, fontweight='bold')
```
Concurrently, inside the lead-plotting loop, an internal text call was executed:
```python
ax.text(start_t - 0.04, base, f"{LEAD_NAMES[l_idx]}", fontsize=9.5, fontweight='bold',
        va='center', ha='right', color=color, zorder=4)
```
Because the X-axis limit was widened to `start_t - 0.32` to accommodate the 1.0 mV calibration pulse, the coordinate `start_t - 0.04` fell *inside* the axis viewport immediately adjacent to the first vertical major grid line ($t = \text{start\_t}$). Consequently, viewers saw the lead name printed twice: once on the Y-axis gutter and once inside the pink grid.

The identical issue was present in Panel B (`ax_ig.text(...)`).

### Implemented Fix
1. Removed all internal `ax.text` and `ax_ig.text` invocations.
2. Assigned territory colors directly to the external Y-axis ticklabels:
```python
ax.set_yticks(offsets)
ax.set_yticklabels([LEAD_NAMES[i] for i in indices], fontsize=9.5, fontweight='bold')
for ticklabel, l_idx in zip(ax.get_yticklabels(), indices):
    t_name = [t for t, info in TERRITORIES.items() if l_idx in info['indices']][0]
    ticklabel.set_color(TERRITORIES[t_name]['color'])
```
This ensures a single, high-contrast, color-coded lead identifier positioned exclusively in the Y-axis margin.

---

## 3. Hardware Profiling: `ptbxl` (CPU) vs. `ptbxl-gpu` (NVIDIA GTX 950M)

### Investigation of User Query
*User Observation*: Would executing figure generation in the `ptbxl-gpu` environment yield a substantial performance boost over the `ptbxl` (CPU) environment?

### Profiling Breakdown
We benchmarked the pipeline across both environments. The total wall-clock time for generating all 5 high-resolution multi-panel figures (300 DPI) is approximately **80–95 seconds** in both environments.

```
Total Execution Time Breakdown per Figure (~17s):
┌────────────────────────────────────────────────────────┬──────────┐
│ Operation                                              │ Duration │
├────────────────────────────────────────────────────────┼──────────┤
│ 1. Forward Pass (SE-ResNet1D prediction)               │ ~0.08 s  │
│ 2. 1D Grad-CAM++ (1 backward gradient pass)           │ ~0.15 s  │
│ 3. Integrated Gradients (50 path interpolation steps)  │ ~0.65 s  │
│ 4. Territory Occlusion Sensitivity (4 branch masks)    │ ~0.12 s  │
│ 5. Matplotlib 300 DPI Vector Rasterization (Agg)       │ ~16.0 s  │
└────────────────────────────────────────────────────────┴──────────┘
```

### Key Technical Findings:
1. **Computational Bottleneck is Software Rasterization, Not Deep Learning**:
   - Neural network inference and gradient attributions (steps 1–4) require $<1.0$ second per record on CPU and $\sim 0.3$ seconds on GPU.
   - Matplotlib's 2D rendering engine (`matplotlib.backends.backend_agg`) is strictly a **single-threaded CPU** software rasterizer written in C++. It renders tens of thousands of pink grid line segments, 18,000 multi-colored `LineCollection` gradient segments, and axes typography entirely on the CPU.
   - Consequently, over **93% of the execution time** is spent in Matplotlib CPU rasterization, making GPU acceleration practically negligible for overall wall-clock time.
2. **CUDA Dynamic Linker Requirement**:
   - In Conda environments where CUDA libraries reside in `$CONDA_PREFIX/lib`, TensorFlow requires `LD_LIBRARY_PATH=/home/awais/anaconda3/envs/ptbxl-gpu/lib:$LD_LIBRARY_PATH` to resolve `libcudart.so.11.0` and `libcudnn.so.8`.
3. **VRAM Constraints on Mobile GPUs**:
   - The NVIDIA GeForce GTX 950M has 4096 MiB total VRAM. By default, TensorFlow's BFC allocator attempts to reserve the entire memory address space, triggering mild allocator fallback warnings (`ran out of memory trying to allocate 4.08GiB`). While non-fatal, managing this on CPU requires zero VRAM allocation overhead.
4. **Conclusion**:
   - Both environments successfully generate the exact same mathematical attributions. For figure rendering, the CPU-bound Matplotlib step dominates regardless of GPU availability.

---

## 4. Verification of Generated Publication Figures

All 5 publication figures were regenerated at 300 DPI:
1. `manuscript/figures/fig3_xai_mi.png` — Record #15647 (100.0% confidence, Antero-Septal / LAD)
2. `manuscript/figures/fig4_xai_sttc.png` — Record #10054 (99.9% confidence, Lateral / LCx)
3. `manuscript/figures/fig5_xai_cd.png` — Record #4893 (100.0% confidence, Antero-Septal / Conduction)
4. `manuscript/figures/fig6_xai_hyp.png` — Record #16182 (99.6% confidence, Lateral / LVH)
5. `manuscript/figures/fig7_xai_norm.png` — Record #2083 (100.0% confidence, Balanced Equilibrium)

All figures strictly adhere to standard clinical ECG grid calibration and contain clean, single lead labels on the Y-axis.

---

## 5. Visual Audit & Standardization of Remaining Sections (B, C, D, and E)

Following user verification of the 2-column layout in Section A, an exhaustive visual audit of Sections B, C, D, and E was conducted across all 5 clinical conditions (MI, STTC, CD, HYP, NORM).

### 5.1 Section B: Axiomatic Integrated Gradients (Lead-Wise Energy)
- **Previous Issues Identified**:
  1. *Horizontal Asynchrony*: Section B was rendered as a single wide 12-lead stacked panel stretching across the full width of both Section A columns. This caused a $2\times$ horizontal time dilation relative to Section A, meaning peaks in Section B did not align vertically with the corresponding ECG waveforms in Section A.
  2. *Vertical Crowding*: Stacking 12 traces in the same vertical space severely compressed the dynamic range of each trace, making smaller lead deflections indistinguishable.
  3. *Left Margin Dead Space*: A 0.28 s blank gap existed between the Y-axis and waveform initiation because calibration pulses are only applicable to Section A.
  4. *Panel Title Collision*: The title `B: Axiomatic Integrated Gradients...` sat too close to the X-axis labels of Section A.
- **Implemented Fixes**:
  1. **Synchronized 2-Column Split**: Section B was converted to an identical 2-column layout (Limb leads $I\text{--}aVF$ on the left, Precordial leads $V_1\text{--}V_6$ on the right).
  2. **1-to-1 Beat Synchronization**: The time scale is now strictly identical to Section A ($25\text{ mm/s}$, with $0.20\text{ s}$ major grid lines). Deflections in Section B now align vertically beat-for-beat with the QRS/ST-T complexes directly above in Section A.
  3. **Invisible Dummy Colorbar Balancing**: Added an invisible dummy colorbar axis to Section B (`cbar_dummy.ax.set_visible(False)`) to ensure the right bounding box of Section B aligns perfectly with the right bounding box of Section A.
  4. **Dynamic Headroom & Continuous Baselines**: Set `ig_spacing = 1.6`, `ylim = (-0.8, 5 * ig_spacing + 2.0)`, and `ig_scale = (0.75 * ig_spacing) / (ig_max + 1e-8)`. This guarantees large voltage attributions (e.g. $V_1$ in MI or $V_5/V_6$ in HYP) never clip the upper border. Added dashed baseline extensions to eliminate dead space.

### 5.2 Section C: Coronary Territory Attribution (Occlusion vs. Attention)
- **Previous Issue Identified**:
  - *Legend Collision*: In cases with dominant vascular territory drops (e.g., $90.9\%$ in STTC, $90.4\%$ in CD, or $79.8\%$ Cavity attention in NORM), tall bars reached into the upper right corner where the legend box (`Territory Occlusion Drop (%)`, `Cross-Territory Attention (%)`) was positioned (`loc='upper right'`).
- **Implemented Fix**:
  - Expanded Y-axis headroom to `ylim(0, max_val * 1.30 + 10.0)`. Even with near-$100\%$ bar values, the legend now sits cleanly above all bars with comfortable whitespace separation.

### 5.3 Section D: Diagnostic Probability Spectrum
- **Previous Issues Identified**:
  - *Text Label Clipping*: With `xlim(0, 105)`, text labels for high-confidence predictions (e.g., `"100.0%"`) overflowed past the right axis border.
  - *Misleading Color Semantics*: The previous code colored `NORM` bright green even when its confidence was $0.1\%$, giving viewers an impression of a positive normal prediction.
- **Implemented Fixes**:
  - Expanded X-axis limits to `xlim(0, 122)`, ensuring 5-character string labels (`100.0%`) always render with clear internal margin.
  - Implemented a clinically coherent **Alert Color Palette**:
    - Ground-truth target pathology is highlighted in crimson red (`#dc2626`) for disease, or emerald green (`#16a34a`) for Normal Sinus Rhythm.
    - Non-target secondary classes are rendered in neutral slate blue (`#94a3b8`) with dark gray normal-weight text labels (`#334155`).

### 5.4 Section E: Electrophysiological Concordance Commentary Box
- **Previous Issue Identified**:
  - The commentary template unconditionally referred to "culprit coronary vascular lesions" and "culprit leads" even for normal control patients (Record #2083).
- **Implemented Fix**:
  - Implemented dynamic, condition-aware commentary generation. Normal control cases now correctly report balanced, non-focal territorial equilibrium ($\Delta P = 0.10$), diffuse non-pathological background activations, and harmonic physiological distribution.

---

## 6. LaTeX Manuscript Sync & Compilation Verification

1. **Updated Captions and In-Text Descriptions**:
   - Standardized captions in `manuscript/main.tex`, `manuscript/main_single.tex`, and `manuscript/4th Draft/` to explicitly describe Panels (A) through (E).
   - Updated in-text case values in Section 6 to match exact percentages rendered in the figures.
2. **MiKTeX / pdfLaTeX Compilation**:
   - Both `main.tex` and `main_single.tex` compiled with **exit code 0** (20 pages and 24 pages respectively).
   - Visual inspection of all regenerated PNGs confirms complete resolution of all identified visual issues.

