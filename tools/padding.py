import torch
import torch.nn as nn

array = torch.Tensor([[1, 2, 3, 2, 1]])

print(array, array.shape)

print(nn.functional.pad(array, [0, 0, 2, 2]))