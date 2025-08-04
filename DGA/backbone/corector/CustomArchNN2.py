from collections import OrderedDict
from typing import Final
import torch
from torch import nn, Tensor

class CustomNN_MArch(nn.Module):
    fun_name:Final[str]

    def __init__(self, sizes:tuple[list[int],list[int]],
                 activations:tuple[list[type[nn.Module]],list[type[nn.Module]]]|type[nn.Module]|None = None):
        super().__init__() # type: ignore
        assert len(sizes) == 2, 'sizes format not correct'
        assert activations is not list or len(activations) == 2, 'activations format not correct'

        act:type[nn.Module]|None

        sizesCommon = sizes[0]
        sizesSpecific = sizes[1]

        if activations is None:
            act = nn.ReLU
        elif isinstance(activations, type):
            act = activations
        else:
            act = None
            if len(activations) != len(sizesSpecific) -2 + len(sizesCommon) -2:
                raise Exception('the number of activation functions and the number of layers do not correspond')

        #self.flatten = nn.Flatten()

        self.insize = sizesCommon[0]//2

        modules:OrderedDict[str,nn.Module] = OrderedDict()

        self.fun_name=f'CustomNN_MArch-layers_{sizesCommon[0]}'
        for i in range(len(sizesCommon)-2):
            modules['layer {}'.format(i)] = nn.Linear(sizesCommon[i], sizesCommon[i+1])
            act_lay:type[nn.Module] = act if act != None else activations[i] # type: ignore
            modules['activ {}'.format(i)] = act_lay()
            self.fun_name+=f'_{act_lay.__name__}_{sizesCommon[i+1]}'
        

        modules['exit layer'] = nn.Linear(sizesCommon[-2], sizesCommon[-1])
        self.fun_name+=f'_{sizesCommon[-1]}'


        self.firstStack = nn.Sequential(modules)

        modules_p:OrderedDict[str,nn.Module] = OrderedDict()
        modules_y:OrderedDict[str,nn.Module] = OrderedDict()

        self.fun_name+=f'_to_{sizesSpecific[0]}'
        for i in range(len(sizesSpecific)-2):
            modules_p['layer {}'.format(i)] = nn.Linear(sizesSpecific[i], sizesSpecific[i+1])
            modules_y['layer {}'.format(i)] = nn.Linear(sizesSpecific[i], sizesSpecific[i+1])
            act_lay:type[nn.Module] = act if act != None else activations[i] # type: ignore
            modules_p['activ {}'.format(i)] = act_lay()
            modules_y['activ {}'.format(i)] = act_lay()
            self.fun_name+=f'_{act_lay.__name__}_{sizesSpecific[i+1]}'

        modules_p['exit layer'] = nn.Linear(sizesSpecific[-2], sizesSpecific[-1])
        modules_y['exit layer'] = nn.Linear(sizesSpecific[-2], sizesSpecific[-1])

        self.pStack = nn.Sequential(modules_p)
        self.yStack = nn.Sequential(modules_y)
        self.fun_name+=f'_{sizesSpecific[-1]}_out_{2*sizesSpecific[-1]}'


    def forward(self, x:Tensor) -> Tensor:
        #x = self.flatten(x)
        r = self.firstStack(x)
        p=torch.cat((x[:,:self.insize],r),dim=1)
        y=torch.cat((x[:,self.insize:],r),dim=1)
        p = self.pStack(p)
        y = self.yStack(y)
        return torch.cat((p,y),dim=1)


