"""
Example code for training a student model using Proximal Policy Distillation (PPD) on Atari environments.
Distillation is performed onto *larger* student policies

"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

try:
    from sb3_distill import ProximalPolicyDistillation, StudentDistill, TeacherDistill
except ImportError:
    import sys
    sys.path.append('../')
    from sb3_distill import ProximalPolicyDistillation, StudentDistill, TeacherDistill

from common import largenet_policy_kwargs


distill_method = 'PPD'
# distill_method = 'student-distill'
# distill_method = 'teacher-distill'

atari_env = "DemonAttack"


# Baseline scores from (Badia et al., 2020)
RANDOM_SCORES = {
    'Atlantis': 12850,
    'BeamRider': 363.9,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152,
    'Enduro': 0,
    'Freeway': 0.0,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'Qbert': 163.9,
    'Seaquest': 68.4,
    'Zaxxon': 32.5}

HUMAN_SCORES = {
    'Atlantis': 29028.1,
    'BeamRider': 16926,
    'CrazyClimber': 35829,
    'DemonAttack': 1971,
    'Enduro': 860.5,
    'Freeway': 29.6,
    'MsPacman': 6951,
    'Pong': 14.6,
    'Qbert': 13455.0,
    'Seaquest': 42054.7,
    'Zaxxon': 9173.3}


def normalize_score(score, env_name):
    return (score - RANDOM_SCORES[env_name])/(HUMAN_SCORES[env_name] - RANDOM_SCORES[env_name])
# End Baseline scores


def lambda_schedule(initial_value, final_value, total_timesteps):
    # TODO: make sure the `timestep' argument is/is not reset when `learn' is called repeatedly!
    def func(timestep):
        """
        Anneal the parameter linearly from `initial_value' to `final_value' over `total_timesteps' timesteps.
        """
        return (final_value - initial_value) / total_timesteps * min(timestep, total_timesteps) + initial_value

    return func


if __name__ == "__main__":
    env = make_atari_env(f"{atari_env}NoFrameskip-v4",
                         n_envs=16,
                         seed=100,
                         vec_env_cls=SubprocVecEnv,
                         wrapper_kwargs=dict(clip_reward=False, terminal_on_life_loss=True))
    env = VecNormalize(env, norm_obs=False)
    env = VecFrameStack(env, n_stack=4)

    teacher_model = PPO.load(f'nn/{atari_env}_teacher_model.ckpt', env=env)

    distill_timesteps = 2_000_000

    if distill_method == 'PPD':
        print('Using PPD!\n\n')

        student_model = ProximalPolicyDistillation("CnnPolicy",
                                                   env,
                                                   verbose=1,
                                                   policy_kwargs=largenet_policy_kwargs,
                                                   n_steps=256,
                                                   batch_size=512,
                                                   n_epochs=4,
                                                   learning_rate=3e-4,
                                                   gamma=0.995,
                                                   ent_coef=0.01,
                                                   tensorboard_log="tensorboard/")

        # Constant distillation-loss weight
        student_model.set_teacher(teacher_model, distill_lambda=2)

        # Linearly annealing distillation-loss weight
        # student_model.set_teacher(teacher_model, distill_lambda=lambda_schedule(3.0, 0.001, 1000000))

    elif distill_method == 'student-distill':
        print('Using student-distill!\n\n')

        student_model = StudentDistill("CnnPolicy",
                                       env,
                                       verbose=1,
                                       policy_kwargs=largenet_policy_kwargs,
                                       n_steps=5,
                                       learning_rate=3e-4,
                                       gamma=0.999,
                                       ent_coef=0.01,
                                       tensorboard_log="tensorboard/")
        student_model.set_teacher(teacher_model)

    elif distill_method == 'teacher-distill':
        print('Using teacher-distill!\n\n')

        n_steps = 5

        student_model = TeacherDistill("CnnPolicy",
                                       env,
                                       verbose=1,
                                       policy_kwargs=largenet_policy_kwargs,
                                       n_steps=n_steps,
                                       learning_rate=3e-4,
                                       gamma=0.999,
                                       ent_coef=0.01,
                                       tensorboard_log="tensorboard/")
        student_model.set_teacher(teacher_model)

    student_model.learn(total_timesteps=distill_timesteps, tb_log_name=atari_env+'_'+distill_method)
    student_model.save(f'nn/{atari_env}_student_'+distill_method+'.ckpt')

    mean_reward, std_reward = evaluate_policy(teacher_model,
                                              teacher_model.get_env(),
                                              n_eval_episodes=10,
                                              deterministic=True)
    print('Raw teacher score: ', mean_reward, '+-', std_reward)
    print('Human-normalized mean teacher score: ', normalize_score(mean_reward, atari_env))

    mean_reward, std_reward = evaluate_policy(student_model,
                                              student_model.get_env(),
                                              n_eval_episodes=10,
                                              deterministic=True)
    print('Raw student reward: ', mean_reward, '+-', std_reward)
    print('Human-normalized mean student score: ', normalize_score(mean_reward, atari_env))
