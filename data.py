import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from visualizer import Visualizer

class Batcher:
    def __init__(self) -> None:
        self.visualizer = Visualizer()
        self.maskFunction = self.twoCircles

    def draw(self, data, target):
        self.visualizer.draw(data, target, self.maskFunction)

    def setMaskFunction(self, maskFunction):
        self.maskFunction = maskFunction

    def getBatch(self, size: int) -> list[torch.Tensor, torch.Tensor]:
        # create torch tensor of size (size, 2)
        data = torch.zeros(size, 2, dtype=torch.float)
        # fill tensor with random values between -1 and 1
        data.uniform_(-1, 1)
        # create target tensor
        target = torch.zeros(size, dtype=torch.float)
        # fill target tensor with 1 if x^2 + y^2 < 1 else 0
        # target = torch.where(data.pow(2).sum(1) < 0.62, torch.ones(size), torch.zeros(size))

        target = torch.where(self.maskFunction(data), torch.ones(size), torch.zeros(size))

        # convert to long
        target = target.long()
        data = data.float()
        return [data, target]

    def twoCircles(self, data):
        return data.pow(2).sum(1) < 0.62