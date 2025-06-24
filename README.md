

# SB3-Distill: policy distillation for stable-baselines3

Here we provide a basic implementation of three methods for performing policy distillation, for use in particular within the stable-baselines3 framework.


## Installation

```bash
python -m pip install git+https://github.com/spiglerg/sb3_distill.git
```

or

```bash
git clone https://github.com/spiglerg/sb3_distill.git
cd sb3_distill
python -m pip install -e .
```

## Example

Train a teacher model using `examples/train_teacher.py`, then distill it onto a new student policy with different architecture (larger network) using `examples/example_distillation.py`.


## Basic usage

SB3-Distill implements distillation algorithms by extending existing DRL algorithms from sb3. For example, PPD is built on, and uses the same arguments as, the PPO implementation of stable-baselines3.

```python
from sb3_distill import ProximalPolicyDistillation

env = [...]
teacher_model = PPO.load('teacher_model.ckpt', env=env)

student_model = ProximalPolicyDistillation("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs, n_steps=128, batch_size=64, n_epochs=5, learning_rate=2.5e-4, gamma=0.999, ent_coef=0.01, tensorboard_log="tensorboard/")

student_model.set_teacher(teacher_model, distill_lambda=1)

student_model.learn(total_timesteps=train_timesteps)
student_model.save('distilled_model.ckpt')
```


## Policy distillation algorithms

We provide basic implementations for the following three methods:

| Method                               | sb3_distill class                  |
| :-------------:                      |:-------------:                     |
| Proximal Policy Distillation <br/> (PPD)   | `ppd.ProximalPolicyDistillation`   |
| Student-distill                      | `student_distill.StudentDistill`   |
| Teacher-distill                      | `teacher_distill.TeacherDistill`   |




## Citing the Project

To cite this repository in publications:

```bibtex
@article{spigler2025ppd,
  author={Giacomo Spigler},
  title={Proximal Policy Distillation},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=WfVXe88oMh}
}
```




