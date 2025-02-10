from collections import OrderedDict
import os
from typing import List, Tuple
import torch as tc
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import CorectionUtilities as cu


def train():
    pass

def test():
    pass



class CNN1(nn.Module):
    

    def __init__(self, sizes:List[int], activations:List[type]|tc.nn.Module|None = None):
        super().__init__()

        act:tc.nn.Module|None = None

        if activations is not None and len(activations) != len(sizes) - 1:
            raise Exception('activations')
        elif activations is None:
            act = nn.ReLU
        elif activations is tc.nn.Module:
            act = activations
        else:
            raise Exception('?')

        self.flatten = nn.Flatten()

        modules = OrderedDict()

        for i in range(len(sizes)-1):
            modules['layer {}'.format(i)] = nn.Linear(sizes[i], sizes[i+1])
            modules['activ {}'.format(i)] = activations[i]() if act is None else act()

        modules['exit layer'] = nn.Linear(sizes[-1], 1)


        self.linear_relu_stack = nn.Sequential(modules)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def default(): return CNN1([5,10])