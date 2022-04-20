import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def mock_image_dataloader(
    image_dimensionality: Tuple[int],
    num_classes: int,
    batch_size: int,
    max_iters: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    if max_iters is None:
        max_iters = batch_size
    X = torch.rand((max_iters, *image_dimensionality))
    T = (torch.rand(max_iters) * num_classes).long()
    dataset = torch.utils.data.TensorDataset(X, T)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
