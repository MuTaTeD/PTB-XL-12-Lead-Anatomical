"""
anatomical_classifier.py
Anatomical Multi-Branch 1D Squeeze-and-Excitation ResNet Classifier for 12-Lead ECGs.

Partitions 12-lead ECG signals into 4 distinct anatomical cardiac territories:
1. Inferior Wall    : Leads [II, III, aVF] (Indices: 1, 2, 5)
2. Antero-Septal Wall: Leads [V1, V2, V3, V4] (Indices: 6, 7, 8, 9)
3. Lateral Wall     : Leads [I, aVL, V5, V6] (Indices: 0, 4, 10, 11)
4. Cavity Reciprocal: Lead  [aVR] (Index: 3)

Features Cross-Territory Squeeze-and-Excitation (SE) Fusion to dynamically weight feedback
from affected anatomical walls before multi-label diagnosis classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def se_block_1d(input_tensor, ratio=8, name="se"):
    """
    Standard Channel Squeeze-and-Excitation block for 1D feature maps.
    """
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D(name=f"{name}_squeeze")(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu', name=f"{name}_ex_1")(se)
    se = layers.Dense(filters, activation='sigmoid', name=f"{name}_ex_2")(se)
    se = layers.Reshape((1, filters), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([input_tensor, se])


def anatomical_branch(x, branch_name, filters=[32, 64, 128], spatial_dropout=0.08, l2_weight=2e-5):
    """
    Dedicated Residual 1D Feature Extraction Branch for a specific anatomical territory.
    """
    # Block 1
    x = layers.Conv1D(filters[0], kernel_size=7, padding='same',
                      kernel_regularizer=regularizers.l2(l2_weight),
                      name=f"{branch_name}_c1")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn1")(x)
    x = layers.ReLU(name=f"{branch_name}_relu1")(x)
    x = layers.SpatialDropout1D(spatial_dropout, name=f"{branch_name}_drop1")(x)
    x = layers.MaxPool1D(pool_size=2, name=f"{branch_name}_pool1")(x)

    # Block 2
    res = x
    x = layers.Conv1D(filters[1], kernel_size=5, padding='same',
                      kernel_regularizer=regularizers.l2(l2_weight),
                      name=f"{branch_name}_c2a")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn2a")(x)
    x = layers.ReLU(name=f"{branch_name}_relu2a")(x)
    x = layers.Conv1D(filters[1], kernel_size=5, padding='same',
                      kernel_regularizer=regularizers.l2(l2_weight),
                      name=f"{branch_name}_c2b")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn2b")(x)
    
    # Projection shortcut if channels change
    if res.shape[-1] != filters[1]:
        res = layers.Conv1D(filters[1], kernel_size=1, padding='same',
                            kernel_regularizer=regularizers.l2(l2_weight),
                            name=f"{branch_name}_proj2")(res)
    x = layers.Add(name=f"{branch_name}_add2")([x, res])
    x = layers.ReLU(name=f"{branch_name}_relu2b")(x)
    x = layers.MaxPool1D(pool_size=2, name=f"{branch_name}_pool2")(x)

    # Block 3 + SE
    res = x
    x = layers.Conv1D(filters[2], kernel_size=3, padding='same',
                      kernel_regularizer=regularizers.l2(l2_weight),
                      name=f"{branch_name}_c3a")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn3a")(x)
    x = layers.ReLU(name=f"{branch_name}_relu3a")(x)
    x = layers.Conv1D(filters[2], kernel_size=3, padding='same',
                      kernel_regularizer=regularizers.l2(l2_weight),
                      name=f"{branch_name}_c3b")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn3b")(x)
    
    if res.shape[-1] != filters[2]:
        res = layers.Conv1D(filters[2], kernel_size=1, padding='same',
                            kernel_regularizer=regularizers.l2(l2_weight),
                            name=f"{branch_name}_proj3")(res)
    x = layers.Add(name=f"{branch_name}_add3")([x, res])
    x = layers.ReLU(name=f"{branch_name}_relu3b")(x)
    x = se_block_1d(x, ratio=8, name=f"{branch_name}_se")
    
    # Global Pooling to obtain territory vector
    branch_feat = layers.GlobalAveragePooling1D(name=f"{branch_name}_gap")(x)
    return branch_feat  # Shape: (batch, 128)


def build_anatomical_se_ecg_classifier(
    input_shape=(1000, 12),
    num_classes=5,
    final_activation='sigmoid',
    dense_dropout=0.3,
    spatial_dropout=0.08,
    l2_weight=2e-5
):
    """
    Builds the Anatomical Multi-Branch SE-ResNet1D Classifier.
    """
    inputs = layers.Input(shape=input_shape, name="ecg_input")

    # Slice 12-lead ECG into 4 Anatomical Territories based on WFDB lead order:
    # 0: I, 1: II, 2: III, 3: AVR, 4: AVL, 5: AVF, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
    inferior_leads = tf.gather(inputs, [1, 2, 5], axis=-1, name="gather_inferior")
    septal_leads = tf.gather(inputs, [6, 7, 8, 9], axis=-1, name="gather_septal")
    lateral_leads = tf.gather(inputs, [0, 4, 10, 11], axis=-1, name="gather_lateral")
    cavity_leads = tf.gather(inputs, [3], axis=-1, name="gather_cavity")

    # Pass each territory through its dedicated anatomical 1D Conv Branch
    feat_inferior = anatomical_branch(inferior_leads, "inferior", spatial_dropout=spatial_dropout, l2_weight=l2_weight)
    feat_septal = anatomical_branch(septal_leads, "septal", spatial_dropout=spatial_dropout, l2_weight=l2_weight)
    feat_lateral = anatomical_branch(lateral_leads, "lateral", spatial_dropout=spatial_dropout, l2_weight=l2_weight)
    feat_cavity = anatomical_branch(cavity_leads, "cavity", spatial_dropout=spatial_dropout, l2_weight=l2_weight)

    # Stack regional feature vectors -> Shape: (batch, 4, 128)
    stacked_territories = layers.Lambda(
        lambda feats: tf.stack(feats, axis=1),
        name="stack_territories"
    )([feat_inferior, feat_septal, feat_lateral, feat_cavity])

    # Cross-Territory Squeeze-and-Excitation Attention
    # Squeeze each 128-dim territory vector to a scalar rating per territory
    sq = layers.GlobalAveragePooling1D(name="territory_squeeze")(stacked_territories)  # (batch, 128)
    ex = layers.Dense(16, activation='relu', name="territory_ex1")(sq)
    ex = layers.Dense(4, activation='softmax', name="territory_softmax_attention")(ex)  # (batch, 4)
    territory_weights = layers.Reshape((4, 1), name="territory_weights_reshape")(ex)

    # Re-weight territory feature vectors dynamically based on cross-territory attention
    weighted_territories = layers.Multiply(name="apply_territory_attention")([stacked_territories, territory_weights])

    # Flatten & Dense Classification Head
    flattened = layers.Flatten(name="flatten_territories")(weighted_territories)
    x = layers.Dropout(dense_dropout, name="head_dropout")(flattened)
    
    outputs = layers.Dense(
        num_classes,
        activation=final_activation,
        kernel_regularizer=regularizers.l2(l2_weight),
        name="classification_head"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Anatomical_SE_ResNet1D")
    return model
