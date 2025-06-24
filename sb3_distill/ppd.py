from typing import Callable
import numpy as np
from gymnasium import spaces
import torch as th
from torch.nn import functional as F
from stable_baselines3.common import distributions
from stable_baselines3.common.utils import explained_variance
from stable_baselines3 import PPO

from sb3_distill.core import PolicyDistillationAlgorithm


class ProximalPolicyDistillation(PolicyDistillationAlgorithm, PPO):
    """

    Proximal Policy Distillation (PPD) algorithm, based on the stable-baselines3 implementation of PPO.

    Paper: https://openreview.net/forum?id=WfVXe88oMh

    Usage:
        model = ProximalPolicyDistillation(usual ppo arguments)
        model.set_teacher(teacher_model, kl_mode="forward", distill_lambda=1.0)

        distill_lambda can be either a floating point or a function. If it is a function, it must take
        a `timestep' argument with the number of elapsed timesteps.

    Most of the code is adapted from sb3's PPO implementation, version 2.7.0a0, but it should be compatible
    with most versions of the sb3 library.
    Parts of the code that differ from the original implementation are marked by "## MODIFIED ##" comments.
    """

    def set_teacher(self, teacher_model, distill_lambda=1.0, kl_mode='forward'):
        """
        Specify or replace teacher model to use for policy distillation.
        ProximalPolicyDistillation will create a separate policy for the student.

        :param teacher_model: SB3 [On/Off]PolicyAlgorithm object to use as teacher for distillation.
        :param distill_lambda: Coefficient of the distillation loss, to balance the student-rewards-based PPO loss.
        :param kl_mode: type of KL divergence to use. "forward" KL(teacher || student) (mean-seeking, mass-covering)
                        or "reverse" KL(student || teacher) (mode-seeking).
        """
        assert kl_mode in ['forward', 'reverse'], \
               f'kl_mode should be either "forward" or "reverse"; invalid option {kl_mode}'

        super().set_teacher(teacher_model=teacher_model)
        self.distill_lambda = distill_lambda
        self.kl_mode = kl_mode

    def train(self) -> None:
        """
        The train() method of PPO is overridden to add the PPD loss. Please note that only a small part of the method
        has been modified. Parts that have been changed are marked as ### MODIFIED ####.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        ### MODIFIED ###
        epoch_distillation_lambda = 0.0
        distillation_losses = []
        ################

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())


                ### MODIFIED ###
                lambda_ = self.distill_lambda
                if isinstance(lambda_, Callable):
                    lambda_ = lambda_(self.num_timesteps)
                    # print(self.num_timesteps, lambda_)
                epoch_distillation_lambda = lambda_

                if hasattr(self, 'teacher_model') and self.teacher_model is not None and lambda_>0.0:
                    teacher_act_distribution = self.teacher_model.policy.get_distribution(rollout_data.observations)
                    student_act_distribution = self.policy.get_distribution(rollout_data.observations)

                    if self.kl_mode == 'forward':
                        # Trying to copy the teacher; mean-seeking
                        kl_divergence = distributions.kl_divergence(teacher_act_distribution, student_act_distribution)
                    elif self.kl_mode == 'reverse':
                        # Trying to find the most probable action of the teacher; mode-seeking
                        kl_divergence = distributions.kl_divergence(student_act_distribution, teacher_act_distribution)

                    if isinstance(teacher_act_distribution,
                                  (distributions.DiagGaussianDistribution,
                                   distributions.StateDependentNoiseDistribution)):
                        kl_divergence = distributions.sum_independent_dims(kl_divergence)

                    # Clipped ratio: note that both ratio (clipped or unclipped) and KL are always >=0;
                    # thus, contrary to the clip on the ratio*advantage, we do not have problems with changing signs.
                    # max(r*kl, clip_r*kl) = max(r, clip(r, 1-e, 1+e))*kl = max(r, 1-e)*kl
                    # clipped_ratio = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    clipped_ratio = th.clamp(ratio, 1-clip_range, None)
                    distillation_loss = th.mean(clipped_ratio * kl_divergence)   # 'ratio' or ''

                    loss = policy_loss \
                           + self.ent_coef * entropy_loss \
                           + self.vf_coef * value_loss \
                           + lambda_ * distillation_loss

                    distillation_losses.append(lambda_*distillation_loss.item())
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                ################


                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        ### MODIFIED ###
        self.logger.record("train/distillation_lambda", epoch_distillation_lambda)
        self.logger.record("train/distillation_loss", np.mean(distillation_losses))
        ################
