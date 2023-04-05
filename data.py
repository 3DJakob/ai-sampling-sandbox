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

    def centerCircle(self, data):
        return data.pow(2).sum(1) < 0.62
    
    def centerSquare(self, data):
        return torch.abs(data).sum(1) < 0.62
    
    def twoCircles(self, data):
        sampleCount = data.shape[0]
        # one cicle at (0.5, 0.5) and one at (-0.5, -0.5)
        # offset tensor sampleCount x 2
        offset = torch.zeros(sampleCount, 2, dtype=torch.float)
        offset[:, 0] = 0.5
        offset[:, 1] = 0.5

        radius = 0.35

        return torch.logical_or((data - offset).pow(2).sum(1) < radius, (data + offset).pow(2).sum(1) < radius)
    
    def twoArchs(self, data):
        sampleCount = data.shape[0]

        offsetX = 0.1
        offsetY = 0.25

        arch1 = generate_half_arch(data, -offsetX, offsetY, flipped=True)
        arch2 = generate_half_arch(data, offsetX, -offsetY, flipped=False)

        return torch.logical_or(arch1, arch2)

def generate_half_arch(data, offsetX, offsetY, flipped=False):
    offset = torch.zeros_like(data)
    offset[:,0] = offsetX
    offset[:,1] = offsetY
    
    outer_circle = (data - offset).pow(2).sum(1) <= 0.25

    inner_circle = (data - offset).pow(2).sum(1) >= 0.1

    doughnut = torch.logical_and(outer_circle, inner_circle)

    half_arch = torch.logical_and(doughnut, (data[:, 0] <= 0 + offsetX) ^ flipped)
    return half_arch