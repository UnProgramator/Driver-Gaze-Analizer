from collections import OrderedDict
from typing import Final, Self, override
from torch import nn, Tensor
import torch

from SimpleNN import SimpleNN


class ComposedNN(SimpleNN):

    pretrainedModule:SimpleNN

    def __init__(self, pretrainedModule:SimpleNN, sizes:list[int], activations:list[type[nn.Module]]|type[nn.Module]|None = None):
        super().__init__(sizes, activations) # type: ignore
        self.pretrainedModule=pretrainedModule
        self.fun_name += f'-pretrainedwith_{self.pretrainedModule.fun_name}'

    @override
    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.pretrainedModule.train(False)
        return self

    @override
    def forward(self, x:Tensor) -> Tensor:
        y = self.pretrainedModule(x)
        r = torch.cat((y,x),dim=1)
        r = super().forward(r)
        return r
