from collections import OrderedDict
from typing import Final
from torch import nn, Tensor


def train():
    pass

def test():
    pass



class SimpleNN(nn.Module):
    
    fun_name:Final[str]

    def __init__(self, sizes:list[int], activations:list[type[nn.Module]]|type[nn.Module]|None = None):
        super().__init__() # type: ignore

        act:type[nn.Module]|None

        if activations is None:
            act = nn.ReLU
        elif isinstance(activations, type):
            act = activations
        else:
            act = None
            if len(activations) != len(sizes) - 2:
                raise Exception('the number of activation functions and the number of layers do not correspond')

        self.flatten = nn.Flatten()

        modules:OrderedDict[str,nn.Module] = OrderedDict()

        self.fun_name=f'SimpleNN-layers_{sizes[0]}'
        for i in range(len(sizes)-2):
            modules['layer {}'.format(i)] = nn.Linear(sizes[i], sizes[i+1])
            act_lay:type[nn.Module] = act if act != None else activations[i] # type: ignore
            modules['activ {}'.format(i)] = act_lay()
            self.fun_name+=f'_{act_lay.__name__}_{sizes[i+1]}'
        

        modules['exit layer'] = nn.Linear(sizes[-2], sizes[-1])
        self.fun_name+=f'_{sizes[-1]}'


        self.linear_relu_stack = nn.Sequential(modules)

    def forward(self, x:Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

    @staticmethod
    def default(): return SimpleNN([5,10])