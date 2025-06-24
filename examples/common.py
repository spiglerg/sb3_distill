from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class IMPALA_Block(nn.Module):
    # Adapted from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    def __init__(self, in_depth, depth):
        super(IMPALA_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding='same')
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding='same')
        self.conv4 = nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        out = self.max1(self.conv1(x))
        out += self.conv3(F.relu(self.conv2(F.relu(out))))
        out += self.conv5(F.relu(self.conv4(F.relu(out))))
        return out


class IMPALA_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, conv_filters=[16, 32, 32], features_dim=256):
        super(IMPALA_CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        self.block1 = IMPALA_Block(n_input_channels, conv_filters[0])
        self.block2 = IMPALA_Block(conv_filters[0], conv_filters[1])
        self.block3 = IMPALA_Block(conv_filters[1], conv_filters[2])

        self.top_part = nn.Sequential(
            nn.Flatten(),
            nn.ReLU()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.forward_blocks(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward_blocks(self, observations: th.Tensor) -> th.Tensor:
        out = observations

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.top_part(out)
        return out

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return F.relu(self.linear(self.forward_blocks(observations)))


conv_filters = [16, 32, 32]
size_hidden_layer = 256

teacher_policy_kwargs = dict(
    features_extractor_class=IMPALA_CNN,
    features_extractor_kwargs=dict(conv_filters=conv_filters, features_dim=size_hidden_layer),
    net_arch=dict(pi=[], vf=[]),
    # activation_fn=nn.ReLU
    optimizer_kwargs=dict(eps=1e-5),
)

largenet_policy_kwargs = dict(
    features_extractor_class=IMPALA_CNN,
    features_extractor_kwargs=dict(conv_filters=[32, 64, 64], features_dim=1024),
    net_arch=[],
    optimizer_kwargs=dict(eps=1e-5),
)
