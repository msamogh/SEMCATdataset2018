import torch 
import torch.nn as nn


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(200, 1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 90),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(90, 5),
            nn.Tanh()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(5, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return out
