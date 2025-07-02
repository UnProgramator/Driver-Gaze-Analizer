from torch import Tensor, nn
import torch

class CustomLoss(nn.Module):
    def __init__(self, error_ok:float):
        super().__init__()
        self.error_ok = error_ok

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        if len(inputs) != len(targets):
            raise ValueError("The length of actual values and predicted values must be the same")
        # Calculate the squared differences (yi - y_b)^2
        squared_diffs = [pow(0 if abs(act - pred) < self.error_ok else (act - pred), 2) for act, pred in zip(inputs, targets)]
        # Calculate the mean of the squared differences
        stepmse = sum(squared_diffs) / len(inputs)
        return stepmse


class CustomLoss_v(nn.Module):
    def __init__(self, error_ok:float):
        super().__init__()
        self.error_ok = error_ok

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        assert len(inputs) == len(targets), "The length of actual values and predicted values must be the same"
        # Calculate the squared differences (yi - y_b)^2

        squared_diffs = torch.abs(inputs - targets)

        squared_diffs[squared_diffs<self.error_ok] = 0

        squared_diffs = torch.pow(squared_diffs,2)
        
        #squared_diffs = [pow(0 if abs(act - pred) < self.error_ok else (act - pred), 2) for act, pred in zip(inputs, targets)]
        # Calculate the mean of the squared differences
        stepmse = torch.sum(squared_diffs) / len(inputs)
        return stepmse


class CustomLossFilter_1(nn.Module):
        def __init__(self):
            super(CustomLossFilter_1, self).__init__()

        def forward(self, inputs, targets) -> Tensor:
            size = len(inputs)
            if len(targets) != size:
                raise ValueError("The length of actual values and predicted values must be the same")
            # Calculate the squared differences (yi - y_b)^2
            squared_diffs = [abs((inputs[i][0] - inp[i][4]) if abs(inp[i][4] - inp[i][2]) < err_ok else (inputs[i][0] - targets[i][0])) for i in range(size)]
            # Calculate the mean of the squared differences
            stepmse = sum(squared_diffs) / size
            return stepmse

class CustomLossFilter_2(nn.Module):
        def __init__(self):
            super(CustomLossFilter_2, self).__init__()

        def forward(self, inputs, targets) -> Tensor:
            size = len(inputs)
            if len(targets) != size:
                raise ValueError("The length of actual values and predicted values must be the same")
            # Calculate the squared differences (yi - y_b)^2
            squared_diffs = [abs(inputs[i][0] - inp[i][4]) if abs(inp[i][4] - targets[i][0]) < err_ok else pow((inputs[i][0] - targets[i][0]),2) for i in range(size)]
            # Calculate the mean of the squared differences
            stepmse = sum(squared_diffs) / size
            return stepmse