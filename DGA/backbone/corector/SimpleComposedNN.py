from collections import OrderedDict
from typing import Final
import torch
from torch import nn, Tensor
from SimpleNN import SimpleNN
from corectors import loadModel


class SimpleComposedNN(SimpleNN):
   
    def __init__(self, module:str, sizes:list[int], activations:list[type[nn.Module]]|type[nn.Module]|None = None):
        super().__init__(sizes=sizes,activations=activations) # type: ignore
        self.preprocessModule:SimpleNN = loadModel(module) #! to change to NamedModule in the future
        self.fun_name += f'-prprocesor_{self.preprocessModule.fun_name.replace('-','_')}'
        
        nn.Sigmoid

    def forward(self, x:Tensor) -> Tensor:
        y = self.preprocessModule(x)
        r = torch.cat((y,x),dim=1)
        return super().forward(x)


