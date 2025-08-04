from collections import OrderedDict
from typing import Final
import torch
from torch import nn, Tensor
from SimpleNN import SimpleNN


class DeciderNN(SimpleNN):
   
    def __init__(self, sizes:list[int], activations:list[type[nn.Module]]|type[nn.Module]|None = None):
        super().__init__(sizes=sizes,activations=activations) # type: ignore
        self.fun_name+=f'_Sigmoid_Rounded_{sizes[-1]}'
        nn.Sigmoid

    def forward(self, x:Tensor) -> Tensor:
        x = super().forward(x)
        return torch.round(torch.sigmoid(x))

