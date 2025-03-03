import torch
from torch import nn


class MPNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(4096 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x, pose_data):
        x = self.state_encoder(x / 255.0)
        z = self.pose_encoder(pose_data)
        return self.out(torch.cat([x, z], dim=1))
