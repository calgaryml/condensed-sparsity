import torch
from torchvision import transforms


class PerImageStandarization(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        mean = torch.mean(x, dim=(-2, -1))
        std = torch.std(x, axis=(-2, -1))
        return transforms.functional.normalize(
            tensor=x, mean=mean, std=std, inplace=self.inplace
        )
