"""
causal_diffusion/diffusion_model.py
Conditional 1D Denoising Diffusion Probabilistic Model (DDPM / DDIM) in TensorFlow/Keras.
"""

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_sinusoidal_time_embedding(timesteps, embedding_dim):
    """
    Sinusoidal positional embeddings for diffusion timesteps t.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb


class DiffusionSchedule:
    """
    Linear or cosine beta variance schedule for DDPM.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = tf.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.alphas_cumprod_prev = tf.concat([[1.0], self.alphas_cumprod[:-1]], axis=0)

        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)


def build_conditional_unet_1d(seq_len=1000, in_channels=12, num_classes=5, time_emb_dim=64):
    """
    1D U-Net Architecture conditioned on diffusion timestep t and diagnostic label y.
    """
    # Inputs
    x_input = layers.Input(shape=(seq_len, in_channels), name='noisy_ecg')
    t_input = layers.Input(shape=(), dtype=tf.int32, name='timestep')
    y_input = layers.Input(shape=(), dtype=tf.int32, name='label')

    # Timestep Embedding
    t_emb = layers.Lambda(lambda t: get_sinusoidal_time_embedding(t, time_emb_dim))(t_input)
    t_emb = layers.Dense(time_emb_dim, activation='swish')(t_emb)

    # Class Label Embedding
    y_emb = layers.Embedding(input_dim=num_classes, output_dim=time_emb_dim)(y_input)
    cond_emb = layers.add([t_emb, y_emb])
    cond_emb = layers.Dense(64, activation='swish')(cond_emb)

    # Stem
    x = layers.Conv1D(64, kernel_size=7, padding='same')(x_input)

    # Encoder Block 1
    c1 = layers.Dense(64)(cond_emb)[:, None, :]
    h1 = layers.add([x, c1])
    h1 = layers.Conv1D(64, kernel_size=5, padding='same', activation='swish')(h1)
    p1 = layers.MaxPooling1D(2)(h1)  # (500, 64)

    # Encoder Block 2
    c2 = layers.Dense(128)(cond_emb)[:, None, :]
    h2 = layers.Conv1D(128, kernel_size=5, padding='same')(p1)
    h2 = layers.add([h2, c2])
    h2 = layers.Activation('swish')(h2)
    p2 = layers.MaxPooling1D(2)(h2)  # (250, 128)

    # Bottleneck with Dilated Conv
    c_mid = layers.Dense(256)(cond_emb)[:, None, :]
    mid = layers.Conv1D(256, kernel_size=5, dilation_rate=2, padding='same')(p2)
    mid = layers.add([mid, c_mid])
    mid = layers.Activation('swish')(mid)
    mid = layers.Conv1D(256, kernel_size=5, dilation_rate=4, padding='same', activation='swish')(mid)

    # Decoder Block 2
    up2 = layers.UpSampling1D(2)(mid)  # (500, 256)
    up2 = layers.Conv1D(128, kernel_size=5, padding='same')(up2)
    cat2 = layers.concatenate([up2, h2])
    d2 = layers.Conv1D(128, kernel_size=5, padding='same', activation='swish')(cat2)

    # Decoder Block 1
    up1 = layers.UpSampling1D(2)(d2)  # (1000, 128)
    up1 = layers.Conv1D(64, kernel_size=5, padding='same')(up1)
    cat1 = layers.concatenate([up1, h1])
    d1 = layers.Conv1D(64, kernel_size=5, padding='same', activation='swish')(cat1)

    # Output: Predicted Noise epsilon_theta
    eps_out = layers.Conv1D(in_channels, kernel_size=5, padding='same', name='eps_pred')(d1)

    model = keras.Model(inputs=[x_input, t_input, y_input], outputs=eps_out, name='Conditional_DDPM_1D')
    return model


class CausalDiffusionEngine:
    """
    Performs DDPM training, DDIM latent abduction (inversion), and counterfactual prediction.
    """
    def __init__(self, model, schedule=None):
        self.model = model
        self.schedule = schedule or DiffusionSchedule(timesteps=1000)

    @tf.function
    def train_step(self, x0, y, optimizer):
        """
        Calculates L_simple(theta) = E [ || eps - eps_theta(x_t, t, y) ||^2 ]
        """
        batch_size = tf.shape(x0)[0]
        t = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.schedule.timesteps, dtype=tf.int32)
        eps = tf.random.normal(shape=tf.shape(x0))

        # Forward noise
        sqrt_alpha = tf.gather(self.schedule.sqrt_alphas_cumprod, t)[:, None, None]
        sqrt_one_minus = tf.gather(self.schedule.sqrt_one_minus_alphas_cumprod, t)[:, None, None]
        xt = sqrt_alpha * x0 + sqrt_one_minus * eps

        with tf.GradientTape() as tape:
            eps_pred = self.model([xt, t, y], training=True)
            loss = tf.reduce_mean(tf.square(eps - eps_pred))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def ddim_invert(self, x0, y_path, num_steps=50):
        """
        Pearl's Step 1 (Abduction): Deterministic forward ODE inversion from x0 to latent noise xT.
        """
        step_stride = self.schedule.timesteps // num_steps
        timesteps = list(range(0, self.schedule.timesteps, step_stride))

        xt = x0
        batch_size = tf.shape(x0)[0]

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_tensor = tf.fill([batch_size], t_cur)
            eps = self.model([xt, t_tensor, y_path], training=False)

            alpha_cur = self.schedule.alphas_cumprod[t_cur]
            alpha_next = self.schedule.alphas_cumprod[t_next]

            # Invert ODE
            x0_pred = (xt - tf.sqrt(1.0 - alpha_cur) * eps) / tf.sqrt(alpha_cur)
            xt = tf.sqrt(alpha_next) * x0_pred + tf.sqrt(1.0 - alpha_next) * eps

        return xt  # Exogenous latent state U_identity

    def ddim_sample(self, xt, y_target, num_steps=50):
        """
        Pearl's Step 3 (Prediction): Deterministic reverse ODE integration from xT to counterfactual.
        """
        step_stride = self.schedule.timesteps // num_steps
        timesteps = list(reversed(range(0, self.schedule.timesteps, step_stride)))

        batch_size = tf.shape(xt)[0]

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_prev = timesteps[i + 1]

            t_tensor = tf.fill([batch_size], t_cur)
            eps = self.model([xt, t_tensor, y_target], training=False)

            alpha_cur = self.schedule.alphas_cumprod[t_cur]
            alpha_prev = self.schedule.alphas_cumprod[t_prev]

            # Tweedie's estimator of clean x0
            x0_pred = (xt - tf.sqrt(1.0 - alpha_cur) * eps) / tf.sqrt(alpha_cur)
            # Reverse ODE step
            xt = tf.sqrt(alpha_prev) * x0_pred + tf.sqrt(1.0 - alpha_prev) * eps

        return xt
