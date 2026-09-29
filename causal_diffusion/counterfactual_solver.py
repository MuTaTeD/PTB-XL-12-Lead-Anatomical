"""
causal_diffusion/counterfactual_solver.py
Minimal Counterfactual Perturbation Solver in TensorFlow.

Solves Equation (3) from the specification:
min_{Delta_X} ||Delta_X||_2^2 + lambda_1 ||Delta_X||_1 + lambda_TV R_TV(Delta_X)
               + gamma * L_clf(X_0 + Delta_X, y_normal)
               + beta * D_physio(X_0 + Delta_X)
               + kappa * D_anat(Delta_X)
"""

import tensorflow as tf
from causal_diffusion.physics import compute_physics_violation_loss, project_signal_to_physics


class CounterfactualOptimizer:
    """
    Finds the minimal, sparse, and physiologically valid counterfactual perturbation Delta_X.
    """
    def __init__(
        self,
        classifier_model,
        target_class_idx=0,      # 0 is NORM
        gamma=20.0,              # Classification loss weight
        beta=15.0,               # Biophysical violation weight
        lambda_l2=1.0,           # L2 energy minimality weight
        lambda_sparse=0.5,       # L1 sparsity weight
        lambda_tv=0.2,           # Total variation smoothness weight
        kappa_anat=10.0,         # Anatomical P-wave preservation weight
        learning_rate=0.01,
        num_iterations=200
    ):
        self.classifier = classifier_model
        self.target_class_idx = target_class_idx
        self.gamma = gamma
        self.beta = beta
        self.lambda_l2 = lambda_l2
        self.lambda_sparse = lambda_sparse
        self.lambda_tv = lambda_tv
        self.kappa_anat = kappa_anat
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def optimize_perturbation(self, x0, anat_mask=None, delta_init=None):
        """
        Solves for Delta_X given factual abnormal ECG x0.
        Args:
            x0: Tensor of shape (1, 1000, 12)
            anat_mask: Binary mask (1, 1000, 12) where 1 indicates P-wave/baseline to preserve.
            delta_init: Optional initial perturbation (e.g. from diffusion DDIM rollout).
        Returns:
            x_cf: Counterfactual ECG (1, 1000, 12)
            delta_x: Perturbation Delta_X (1, 1000, 12)
            history: Dictionary of loss progression
        """
        if delta_init is None:
            delta_x = tf.Variable(tf.zeros_like(x0), trainable=True)
        else:
            delta_x = tf.Variable(delta_init, trainable=True)

        if anat_mask is None:
            anat_mask = tf.zeros_like(x0)
        else:
            anat_mask = tf.cast(anat_mask, tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        history = {'total_loss': [], 'clf_loss': [], 'l2_loss': [], 'physio_loss': [], 'prob_norm': []}

        for step in range(self.num_iterations):
            with tf.GradientTape() as tape:
                # Candidate counterfactual
                x_candidate = x0 + delta_x

                # 1. Classification Loss (driving prediction to y_normal)
                probs = self.classifier(x_candidate, training=False)
                prob_target = probs[:, self.target_class_idx]
                # Log-loss to reach probability ~ 1.0
                clf_loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(prob_target, 1e-7, 1.0)))

                # 2. Minimality & Sparsity
                l2_loss = tf.reduce_mean(tf.square(delta_x))
                l1_loss = tf.reduce_mean(tf.abs(delta_x))

                # 3. Total variation (temporal smoothness)
                tv_loss = tf.reduce_mean(tf.square(delta_x[:, 1:, :] - delta_x[:, :-1, :]))

                # 4. Biophysical constraints (Einthoven + Goldberger)
                physio_loss = compute_physics_violation_loss(x_candidate)

                # 5. Anatomical P-wave invariance
                anat_loss = tf.reduce_mean(tf.square(delta_x * anat_mask))

                # Composite Objective
                total_loss = (
                    self.lambda_l2 * l2_loss +
                    self.lambda_sparse * l1_loss +
                    self.lambda_tv * tv_loss +
                    self.gamma * clf_loss +
                    self.beta * physio_loss +
                    self.kappa_anat * anat_loss
                )

            grads = tape.gradient(total_loss, [delta_x])
            optimizer.apply_gradients(zip(grads, [delta_x]))

            # Project onto exact null-space every 5 steps and at completion
            if step % 5 == 0 or step == self.num_iterations - 1:
                # Ensure x0 + delta_x exactly satisfies physical laws
                x_projected = project_signal_to_physics(x0 + delta_x)
                delta_x.assign(x_projected - x0)

            history['total_loss'].append(float(total_loss.numpy()))
            history['clf_loss'].append(float(clf_loss.numpy()))
            history['l2_loss'].append(float(l2_loss.numpy()))
            history['physio_loss'].append(float(physio_loss.numpy()))
            history['prob_norm'].append(float(prob_target.numpy()[0]))

        x_cf = project_signal_to_physics(x0 + delta_x)
        final_delta = x_cf - x0
        return x_cf, final_delta, history
