from sqlalchemy.sql.elements import True_
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionClassify(nn.Module):
    def __init__(self):
        super(EmotionClassify, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.norm = nn.BatchNorm2d(num_features=10)

        self.fc1 = nn.Linear(in_features=810, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=7)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, U):

        # spatial transformer
        V = self.stn(U)

        out = F.leaky_relu(self.conv1(V))
        out = self.conv2(out)
        out = F.leaky_relu(self.pool2(out))

        out = F.leaky_relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.leaky_relu(self.pool3(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.leaky_relu(self.fc1(out))
        out = self.fc2(out)

        return out
