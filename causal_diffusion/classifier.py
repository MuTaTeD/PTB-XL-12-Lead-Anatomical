"""
causal_diffusion/classifier.py
1D ResNet Diagnostic Classifier for 12-Lead ECGs in TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def squeeze_excitation_1d(x, reduction_ratio=8):
    """
    1D Channel / Lead Squeeze-and-Excitation Attention Module.
    Recovers lead-wise interdependencies and recalibrates channel feature maps.
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    reduced_channels = max(1, channels // reduction_ratio)
    se = layers.Dense(reduced_channels, activation='relu', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
    se = layers.Reshape((1, channels))(se)
    return layers.multiply([x, se])


def se_resnet_block_1d(x, filters, kernel_size=5, stride=1, downsample=False, reduction_ratio=8):
    """
    1D Residual Block integrated with Squeeze-and-Excitation (SE) Lead Attention.
    """
    shortcut = x
    if downsample or stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)

    # Apply Lead-wise Channel Squeeze-and-Excitation Attention
    y = squeeze_excitation_1d(y, reduction_ratio=reduction_ratio)

    out = layers.add([shortcut, y])
    out = layers.ReLU()(out)
    return out


def resnet_block_1d(x, filters, kernel_size=5, stride=1, downsample=False):
    """
    Standard 1D Residual Block with Conv1D, BatchNormalization, and ReLU.
    """
    shortcut = x
    if downsample or stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)

    out = layers.add([shortcut, y])
    out = layers.ReLU()(out)
    return out


def build_ecg_classifier(input_shape=(1000, 12), num_classes=5, final_activation='sigmoid'):
    """
    Builds 1D ResNet classifier for 12-lead ECG signals.
    Input: (batch_size, 1000, 12)
    Output: (batch_size, num_classes) diagnostic logits / probabilities.
    """
    inputs = layers.Input(shape=input_shape, name='ecg_input')

    # Initial Stem
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual Stages
    x = resnet_block_1d(x, filters=32, stride=1)
    x = resnet_block_1d(x, filters=64, stride=2, downsample=True)
    x = resnet_block_1d(x, filters=128, stride=2, downsample=True)
    x = resnet_block_1d(x, filters=256, stride=2, downsample=True)

    # Global Pooling & Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation=final_activation, name='diag_probs')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='ECG_ResNet1D')
    return model


def se_resnet_block_1d_reg(x, filters, kernel_size=5, stride=1, downsample=False, reduction_ratio=8, l2_weight=1e-4):
    """
    L2 Regularized 1D Residual Block with Squeeze-and-Excitation Lead Attention and Spatial Dropout.
    """
    l2_reg = regularizers.l2(l2_weight) if l2_weight > 0 else None
    shortcut = x
    if downsample or stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same', kernel_regularizer=l2_reg)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer=l2_reg)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_regularizer=l2_reg)(y)
    y = layers.BatchNormalization()(y)

    # Lead-wise Channel Squeeze-and-Excitation Attention
    y = squeeze_excitation_1d(y, reduction_ratio=reduction_ratio)

    out = layers.add([shortcut, y])
    out = layers.ReLU()(out)
    out = layers.SpatialDropout1D(0.15)(out)
    return out


def build_regularized_se_ecg_classifier(input_shape=(1000, 12), num_classes=5, final_activation='sigmoid', dropout_rate=0.5, l2_weight=1e-4):
    """
    Builds Regularized SE-ResNet1D Classifier with L2 Weight Decay, Spatial Dropout, and Lead Attention.
    Designed specifically to combat overfitting and maximize out-of-fold generalization.
    """
    l2_reg = regularizers.l2(l2_weight) if l2_weight > 0 else None
    inputs = layers.Input(shape=input_shape, name='ecg_input')

    # 1. Lead-wise Input Attention
    stem_attn = squeeze_excitation_1d(inputs, reduction_ratio=4)

    # 2. Conv Stem with L2 Penalty
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same', kernel_regularizer=l2_reg)(stem_attn)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # 3. Regularized SE Residual Stages
    x = se_resnet_block_1d_reg(x, filters=32, stride=1, reduction_ratio=8, l2_weight=l2_weight)
    x = se_resnet_block_1d_reg(x, filters=64, stride=2, downsample=True, reduction_ratio=8, l2_weight=l2_weight)
    x = se_resnet_block_1d_reg(x, filters=128, stride=2, downsample=True, reduction_ratio=8, l2_weight=l2_weight)
    x = se_resnet_block_1d_reg(x, filters=256, stride=2, downsample=True, reduction_ratio=8, l2_weight=l2_weight)

    # 4. Global Pooling & Strongly Regularized Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation=final_activation, kernel_regularizer=l2_reg, name='diag_probs')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='ECG_Regularized_SE_ResNet1D')
    return model


def se_resnet_block_1d_calibrated(x, filters, kernel_size=5, stride=1, downsample=False, reduction_ratio=8, l2_weight=2e-5, spatial_dropout=0.08):
    """
    Calibrated 1D Residual Block with Squeeze-and-Excitation Lead Attention, L2 Decay (2e-5), and Soft Spatial Dropout (0.08).
    """
    l2_reg = regularizers.l2(l2_weight) if l2_weight > 0 else None
    shortcut = x
    if downsample or stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same', kernel_regularizer=l2_reg)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer=l2_reg)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_regularizer=l2_reg)(y)
    y = layers.BatchNormalization()(y)

    # Lead-wise Channel Squeeze-and-Excitation Attention
    y = squeeze_excitation_1d(y, reduction_ratio=reduction_ratio)

    out = layers.add([shortcut, y])
    out = layers.ReLU()(out)
    if spatial_dropout > 0:
        out = layers.SpatialDropout1D(spatial_dropout)(out)
    return out


def build_calibrated_se_ecg_classifier(input_shape=(1000, 12), num_classes=5, final_activation='sigmoid', dense_dropout=0.3, spatial_dropout=0.08, l2_weight=2e-5):
    """
    Builds Calibrated SE-ResNet1D Classifier with optimal L2 Weight Decay (2e-5), Spatial Dropout (0.08), and Dense Dropout (0.3).
    Maximizes feature representation accuracy while maintaining strong generalization via Cutout augmentation.
    """
    l2_reg = regularizers.l2(l2_weight) if l2_weight > 0 else None
    inputs = layers.Input(shape=input_shape, name='ecg_input')

    # 1. Lead-wise Input Attention
    stem_attn = squeeze_excitation_1d(inputs, reduction_ratio=4)

    # 2. Conv Stem
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same', kernel_regularizer=l2_reg)(stem_attn)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # 3. Calibrated SE Residual Stages
    x = se_resnet_block_1d_calibrated(x, filters=32, stride=1, reduction_ratio=8, l2_weight=l2_weight, spatial_dropout=spatial_dropout)
    x = se_resnet_block_1d_calibrated(x, filters=64, stride=2, downsample=True, reduction_ratio=8, l2_weight=l2_weight, spatial_dropout=spatial_dropout)
    x = se_resnet_block_1d_calibrated(x, filters=128, stride=2, downsample=True, reduction_ratio=8, l2_weight=l2_weight, spatial_dropout=spatial_dropout)
    x = se_resnet_block_1d_calibrated(x, filters=256, stride=2, downsample=True, reduction_ratio=8, l2_weight=l2_weight, spatial_dropout=spatial_dropout)

    # 4. Global Pooling & Calibrated Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(num_classes, activation=final_activation, kernel_regularizer=l2_reg, name='diag_probs')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='ECG_Calibrated_SE_ResNet1D')
    return model




