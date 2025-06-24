import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

from common import teacher_policy_kwargs


atari_env = "DemonAttack"

if __name__ == "__main__":
    env = make_atari_env(f"{atari_env}NoFrameskip-v4",
                         n_envs=16,
                         seed=0,
                         vec_env_cls=SubprocVecEnv,
                         wrapper_kwargs=dict(clip_reward=False, terminal_on_life_loss=True))
    env = VecNormalize(env, norm_obs=False)
    env = VecFrameStack(env, n_stack=4)

    teacher_model = PPO("CnnPolicy",
                        env,
                        verbose=1,
                        policy_kwargs=teacher_policy_kwargs,
                        n_steps=256,
                        batch_size=512,
                        n_epochs=4,
                        learning_rate=3e-4,
                        gamma=0.995,
                        ent_coef=0.01,
                        tensorboard_log="tensorboard/")

    teacher_model.learn(total_timesteps=10_000_000, tb_log_name=f'{atari_env}_teacher_10M')
    teacher_model.save(f'nn/{atari_env}_teacher_model.ckpt')

    mean_reward, std_reward = evaluate_policy(teacher_model, teacher_model.get_env(), n_eval_episodes=10)
    print('Final teacher reward (raw): ', mean_reward, '+-', std_reward)
