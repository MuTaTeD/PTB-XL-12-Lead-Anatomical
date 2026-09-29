"""
unet_1d.py
1D Conditional Diffusion U-Net for 12-Lead Electrocardiogram (ECG) Signals.

Architecture Features:
- 1D Temporal Residual Downsampling & Upsampling Blocks
- Sinusoidal Timestep Embedding (t in [1, T])
- Multi-Label Class Condition Embedding (c in {0,1}^5) supporting Classifier-Free Guidance (CFG)
- Multi-Head Self-Attention Bottleneck
- Preserves (batch_size, 1000, 12) input/output shape matching PTB-XL ECGs
"""

import math
import tensorflow as tf
from tensorflow.keras import layers, regularizers


class SinusoidalPositionalEmbedding(layers.Layer):
    """
    Computes Sinusoidal Timestep Embeddings for scalar diffusion timesteps t.
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, timesteps):
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        timesteps = tf.cast(timesteps, tf.float32)
        emb = tf.cast(timesteps[:, None], tf.float32) * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class ResidualBlock1D(layers.Layer):
    """
    1D Conv Residual Block with Timestep & Class Condition Projection.
    """
    def __init__(self, filters, kernel_size=5, stride=1, downsample=False, upsample=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.downsample = downsample
        self.upsample = upsample

        self.conv1 = layers.Conv1D(filters, kernel_size=kernel_size, padding='same')
        self.conv2 = layers.Conv1D(filters, kernel_size=kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        self.emb_proj = layers.Dense(filters)

        if downsample:
            self.resample = layers.Conv1D(filters, kernel_size=1, strides=2, padding='same')
            self.main_conv = layers.Conv1D(filters, kernel_size=kernel_size, strides=2, padding='same')
        elif upsample:
            self.resample = layers.Conv1DTranspose(filters, kernel_size=1, strides=2, padding='same')
            self.main_conv = layers.Conv1DTranspose(filters, kernel_size=kernel_size, strides=2, padding='same')
        else:
            self.resample = None
            self.main_conv = None

        self.shortcut_conv = layers.Conv1D(filters, kernel_size=1, padding='same')

    def call(self, x, emb=None):
        shortcut = x
        if self.resample is not None:
            shortcut = self.resample(shortcut)
        elif x.shape[-1] != self.filters:
            shortcut = self.shortcut_conv(shortcut)

        if self.main_conv is not None:
            h = self.main_conv(x)
        else:
            h = self.conv1(x)

        h = self.bn1(h)
        h = tf.nn.swish(h)

        if emb is not None:
            emb_out = self.emb_proj(emb)  # (batch, filters)
            emb_out = tf.nn.swish(emb_out)[:, None, :]  # (batch, 1, filters)
            h = h + emb_out

        h = self.conv2(h)
        h = self.bn2(h)

        return tf.nn.swish(h + shortcut)


class MultiHeadAttention1D(layers.Layer):
    """
    1D Temporal Multi-Head Self-Attention Layer for Bottleneck Representations.
    """
    def __init__(self, num_heads=4, key_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.bn = layers.BatchNormalization()

    def call(self, x):
        attn_out = self.mha(query=x, value=x, key=x)
        return self.bn(x + attn_out)


def build_conditional_unet_1d(input_shape=(1000, 12), num_classes=5, base_filters=32, time_emb_dim=256):
    """
    Builds 1D Conditional Diffusion U-Net model.

    Inputs:
    - x_t: Noised ECG signal tensor of shape (batch, 1000, 12)
    - timestep: Scalar integer tensor of shape (batch,)
    - class_label: One-hot / multi-label condition vector of shape (batch, 5)

    Outputs:
    - noise_pred: Predicted noise tensor of shape (batch, 1000, 12)
    """
    # 1. Inputs
    x_input = layers.Input(shape=input_shape, name='noised_ecg_input')
    t_input = layers.Input(shape=(), dtype=tf.int32, name='timestep_input')
    c_input = layers.Input(shape=(num_classes,), dtype=tf.float32, name='class_condition_input')

    # 2. Embeddings
    t_emb = SinusoidalPositionalEmbedding(dim=time_emb_dim)(t_input)
    t_emb = layers.Dense(time_emb_dim, activation=tf.nn.swish)(t_emb)
    t_emb = layers.Dense(time_emb_dim, activation=tf.nn.swish)(t_emb)

    c_emb = layers.Dense(time_emb_dim, activation=tf.nn.swish)(c_input)
    c_emb = layers.Dense(time_emb_dim, activation=tf.nn.swish)(c_emb)

    cond_emb = t_emb + c_emb

    # 3. Encoder Path (Downsampling)
    # Stage 1: 1000 steps, 64 filters
    x1 = ResidualBlock1D(base_filters)(x_input, cond_emb)
    x1 = ResidualBlock1D(base_filters)(x1, cond_emb)

    # Stage 2: 500 steps, 128 filters
    x2 = ResidualBlock1D(base_filters * 2, downsample=True)(x1, cond_emb)
    x2 = ResidualBlock1D(base_filters * 2)(x2, cond_emb)

    # Stage 3: 250 steps, 256 filters
    x3 = ResidualBlock1D(base_filters * 4, downsample=True)(x2, cond_emb)
    x3 = ResidualBlock1D(base_filters * 4)(x3, cond_emb)

    # Stage 4: 125 steps, 512 filters
    x4 = ResidualBlock1D(base_filters * 8, downsample=True)(x3, cond_emb)
    x4 = ResidualBlock1D(base_filters * 8)(x4, cond_emb)

    # 4. Bottleneck with Self-Attention
    bn = MultiHeadAttention1D(num_heads=4, key_dim=64)(x4)
    bn = ResidualBlock1D(base_filters * 8)(bn, cond_emb)

    # 5. Decoder Path (Upsampling with Skip Connections)
    # Stage 4 Up: 125 -> 250 steps
    d4 = layers.Concatenate()([bn, x4])
    d4 = ResidualBlock1D(base_filters * 4, upsample=True)(d4, cond_emb)

    # Stage 3 Up: 250 -> 500 steps
    d3 = layers.Concatenate()([d4, x3])
    d3 = ResidualBlock1D(base_filters * 2, upsample=True)(d3, cond_emb)

    # Stage 2 Up: 500 -> 1000 steps
    d2 = layers.Concatenate()([d3, x2])
    d2 = ResidualBlock1D(base_filters, upsample=True)(d2, cond_emb)

    # Stage 1 Up: 1000 steps
    d1 = layers.Concatenate()([d2, x1])
    d1 = ResidualBlock1D(base_filters)(d1, cond_emb)

    # 6. Output Noise Prediction
    output_noise = layers.Conv1D(input_shape[-1], kernel_size=3, padding='same', name='predicted_noise')(d1)

    model = tf.keras.Model(inputs=[x_input, t_input, c_input], outputs=output_noise, name='Conditional_DDPM_UNet_1D')
    return model
