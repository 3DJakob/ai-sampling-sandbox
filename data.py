import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def getBatch(size: int) -> list[torch.Tensor, torch.Tensor]:
    # create torch tensor of size (size, 2)
    data = torch.zeros(size, 2, dtype=torch.float)
    # fill tensor with random values between -1 and 1
    data.uniform_(-1, 1)
    # create target tensor
    target = torch.zeros(size, dtype=torch.float)
    # fill target tensor with 1 if x^2 + y^2 < 1 else 0
    target = torch.where(data.pow(2).sum(1) < 0.62, torch.ones(size), torch.zeros(size))

    # convert to long
    target = target.long()
    data = data.float()
    return [data, target]

