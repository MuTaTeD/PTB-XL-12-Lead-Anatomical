import os
import subprocess

dot_model_progression = """
digraph G {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor="lightblue", fontname="Arial", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10];
    
    subgraph cluster_model1 {
        label="Model 1: Baseline SE-ResNet1D";
        style=dashed; color=gray;
        M1_in [label="12-Lead ECG (1000x12)"];
        M1_res [label="Shared 1D ResNet Blocks\\n(12 Channels)"];
        M1_se [label="Squeeze-and-Excitation"];
        M1_pool [label="Global Average Pooling"];
        M1_out [label="Diagnostic Output"];
        M1_in -> M1_res -> M1_se -> M1_pool -> M1_out;
    }
    
    subgraph cluster_model2 {
        label="Model 2: Anatomical Multi-Branch SE-ResNet1D";
        style=dashed; color=gray;
        M2_in [label="12-Lead ECG (1000x12)"];
        
        M2_b1 [label="Inferior Branch\\n(II, III, aVF)"];
        M2_b2 [label="Antero-Septal Branch\\n(V1-V4)"];
        M2_b3 [label="Lateral Branch\\n(I, aVL, V5, V6)"];
        M2_b4 [label="Cavity Branch\\n(aVR)"];
        
        M2_r1 [label="1D ResNet"];
        M2_r2 [label="1D ResNet"];
        M2_r3 [label="1D ResNet"];
        M2_r4 [label="1D ResNet"];
        
        M2_concat [label="Concatenate Features"];
        M2_se [label="Squeeze-and-Excitation"];
        M2_pool [label="Global Average Pooling"];
        M2_out [label="Diagnostic Output"];
        
        M2_in -> M2_b1; M2_in -> M2_b2; M2_in -> M2_b3; M2_in -> M2_b4;
        M2_b1 -> M2_r1; M2_b2 -> M2_r2; M2_b3 -> M2_r3; M2_b4 -> M2_r4;
        M2_r1 -> M2_concat; M2_r2 -> M2_concat; M2_r3 -> M2_concat; M2_r4 -> M2_concat;
        M2_concat -> M2_se -> M2_pool -> M2_out;
    }

    subgraph cluster_model3 {
        label="Model 3: Anatomical Territory-Dropout SE-ResNet1D (Proposed)";
        style=dashed; color=gray;
        M3_in [label="12-Lead ECG (1000x12)"];
        
        M3_b1 [label="Inferior Branch"];
        M3_b2 [label="Antero-Septal Branch"];
        M3_b3 [label="Lateral Branch"];
        M3_b4 [label="Cavity Branch"];
        
        M3_r1 [label="1D ResNet"];
        M3_r2 [label="1D ResNet"];
        M3_r3 [label="1D ResNet"];
        M3_r4 [label="1D ResNet"];
        
        M3_d1 [label="Territory Dropout (p=0.15)", fillcolor="lightpink"];
        M3_d2 [label="Territory Dropout (p=0.15)", fillcolor="lightpink"];
        M3_d3 [label="Territory Dropout (p=0.15)", fillcolor="lightpink"];
        M3_d4 [label="Territory Dropout (p=0.15)", fillcolor="lightpink"];
        
        M3_concat [label="Concatenate Features"];
        M3_se [label="Squeeze-and-Excitation"];
        M3_pool [label="Global Average Pooling"];
        M3_out [label="Diagnostic Output"];
        
        M3_in -> M3_b1; M3_in -> M3_b2; M3_in -> M3_b3; M3_in -> M3_b4;
        M3_b1 -> M3_r1 -> M3_d1; 
        M3_b2 -> M3_r2 -> M3_d2; 
        M3_b3 -> M3_r3 -> M3_d3; 
        M3_b4 -> M3_r4 -> M3_d4;
        M3_d1 -> M3_concat; M3_d2 -> M3_concat; M3_d3 -> M3_concat; M3_d4 -> M3_concat;
        M3_concat -> M3_se -> M3_pool -> M3_out;
    }
}
"""

dot_ddpm_unet = """
digraph G {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="lightyellow", fontname="Arial", margin="0.2,0.1"];
    edge [fontname="Arial", fontsize=10];
    
    subgraph cluster_unet {
        label="1D Conditional DDPM U-Net Engine";
        style=dashed; color=gray;
        
        Input [label="Noisy Signal x_t", fillcolor="lightcyan"];
        Time [label="Time Step t\\n(Sinusoidal Emb.)", shape=ellipse, fillcolor="white"];
        Class [label="Target Class y\\n(e.g., NORM)", shape=ellipse, fillcolor="white"];
        
        Enc1 [label="Down Block 1\\n(Conv1D)"];
        Enc2 [label="Down Block 2\\n(Conv1D + Attn)"];
        Enc3 [label="Down Block 3\\n(Conv1D + Attn)"];
        
        Bot [label="Bottleneck\\n(Conv1D + Attn)", fillcolor="lightsalmon"];
        
        Dec3 [label="Up Block 3\\n(Conv1D + Attn)"];
        Dec2 [label="Up Block 2\\n(Conv1D + Attn)"];
        Dec1 [label="Up Block 1\\n(Conv1D)"];
        
        Output [label="Predicted Noise ε_θ", fillcolor="lightgreen"];
        
        Input -> Enc1 -> Enc2 -> Enc3 -> Bot -> Dec3 -> Dec2 -> Dec1 -> Output;
        
        # Skip connections
        Enc1 -> Dec1 [style=dotted, label="Skip"];
        Enc2 -> Dec2 [style=dotted, label="Skip"];
        Enc3 -> Dec3 [style=dotted, label="Skip"];
        
        # Conditioning
        Time -> Enc1 [style=dashed, color=gray];
        Time -> Enc2 [style=dashed, color=gray];
        Time -> Enc3 [style=dashed, color=gray];
        Time -> Bot [style=dashed, color=gray];
        Time -> Dec3 [style=dashed, color=gray];
        Time -> Dec2 [style=dashed, color=gray];
        Time -> Dec1 [style=dashed, color=gray];
        
        Class -> Bot [style=dashed, color=gray, label="Conditioning"];
    }
    
    Guidance [label="Classifier Guidance\\n∇ log p(y|x_t)", shape=note, fillcolor="white"];
    Guidance -> Output [label="Modifies Sampling (DDIM)"];
}
"""

dot_xai_framework = """
digraph G {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="lightblue", fontname="Arial", margin="0.15,0.1"];
    edge [fontname="Arial", fontsize=10];
    
    subgraph cluster_input {
        label="1. Standard 12-Lead ECG Input";
        style=dashed; color=gray;
        ECG_in [label="12-Lead ECG Signal\\n(1000 x 12, 100 Hz)", fillcolor="lightcyan"];
    }
    
    subgraph cluster_decomp {
        label="2. Anatomical Lead Decomposition";
        style=dashed; color=gray;
        T1 [label="Inferior Territory\\n(II, III, aVF)", fillcolor="#ffe0b2"];
        T2 [label="Antero-Septal Territory\\n(V1, V2, V3, V4)", fillcolor="#c8e6c9"];
        T3 [label="Lateral Territory\\n(I, aVL, V5, V6)", fillcolor="#bbdefb"];
        T4 [label="Cavity / Reciprocal\\n(aVR)", fillcolor="#e1bee7"];
    }
    
    ECG_in -> T1;
    ECG_in -> T2;
    ECG_in -> T3;
    ECG_in -> T4;
    
    subgraph cluster_model {
        label="3. Anatomical SE-ResNet1D Model";
        style=dashed; color=gray;
        B1 [label="SE-ResNet Branch 1\\n(Inferior)"];
        B2 [label="SE-ResNet Branch 2\\n(Antero-Septal)"];
        B3 [label="SE-ResNet Branch 3\\n(Lateral)"];
        B4 [label="SE-ResNet Branch 4\\n(Cavity)"];
        
        Concat [label="Concatenate & SE Bottleneck\\nLearned Cross-Territory Softmax Attention", fillcolor="#fff9c4"];
        Classifier [label="Diagnostic Head\\n[NORM, MI, STTC, CD, HYP]", fillcolor="#ffcdd2"];
        
        T1 -> B1; T2 -> B2; T3 -> B3; T4 -> B4;
        B1 -> Concat; B2 -> Concat; B3 -> Concat; B4 -> Concat;
        Concat -> Classifier;
    }
    
    subgraph cluster_xai {
        label="4. Multi-Scale Anatomical Explainability Engine";
        style=dashed; color=navy;
        
        Macro [label="Macro-Tier:\\nTerritory Occlusion (ΔP)\\n& Attention Share w_attn", fillcolor="#ffecb3"];
        Meso [label="Meso-Tier:\\n1D Multi-Branch Grad-CAM++\\n(Morphological Waveform Localization)", fillcolor="#d1c4e9"];
        Micro [label="Micro-Tier:\\nAxiomatic Integrated Gradients\\n(Completeness: |ΣIG - ΔScore| < 0.02)", fillcolor="#b2dfdb"];
    }
    
    Classifier -> Macro [label="Territory Masking"];
    Classifier -> Meso [label="Branch Gradients ∇_A y^c"];
    Classifier -> Micro [label="Path Integral ∫ ∇_x F"];
    
    subgraph cluster_output {
        label="5. Clinician Decision Support";
        style=dashed; color=darkgreen;
        Viz [label="High-Resolution 3.0s Zoom Overlay\\nClinical Pink ECG Grid (0.20s/0.04s)\\nMulti-Panel Attribution Profile", fillcolor="#f8bbd0", shape=ellipse];
    }
    
    Macro -> Viz;
    Meso -> Viz;
    Micro -> Viz;
}
"""

with open('model_progression.dot', 'w') as f:
    f.write(dot_model_progression)

with open('ddpm_unet.dot', 'w') as f:
    f.write(dot_ddpm_unet)

with open('xai_framework.dot', 'w') as f:
    f.write(dot_xai_framework)

subprocess.run(["dot", "-Tpng", "model_progression.dot", "-o", "manuscript/figures/fig_models.png"])
subprocess.run(["dot", "-Tpng", "xai_framework.dot", "-o", "manuscript/figures/fig_xai_framework.png"])
print("Diagrams generated successfully.")

