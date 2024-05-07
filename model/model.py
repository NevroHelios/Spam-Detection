import torch.nn as nn

class DetectSpamV0(nn.Module):
    """Model Architecture"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=3000, out_features=2000)
        self.layer2 = nn.Linear(in_features=2000, out_features=1000)
        self.layer3 = nn.Linear(in_features=1000, out_features=100)
        self.layer4 = nn.Linear(in_features=100, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))
  