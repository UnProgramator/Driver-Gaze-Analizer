from torch import Tensor, nn

class CustomLoss(nn.Module):
    def __init__(self, error_ok):
        super().__init__()
        self.error_ok = error_ok

    def forward(self, inputs, targets):
        if len(inputs) != len(targets):
            raise ValueError("The length of actual values and predicted values must be the same")
        # Calculate the squared differences (yi - y_b)^2
        squared_diffs = [pow(0 if abs(act - pred) < self.error_ok else (act - pred), 2) for act, pred in zip(inputs, targets)]
        # Calculate the mean of the squared differences
        stepmse = sum(squared_diffs) / len(inputs)
        return stepmse


class CustomLoss_v(nn.Module):
    def __init__(self, error_ok):
        super().__init__()
        self.error_ok = error_ok

    def forward(self, inputs:Tensor, targets:Tensor):
        if len(inputs) != len(targets):
            raise ValueError("The length of actual values and predicted values must be the same")
        # Calculate the squared differences (yi - y_b)^2
        
        squared_diffs = inputs - targets

        squared_diffs[squared_diffs<self.error_ok] = 0

        squared_diffs = pow(squared_diffs, 2)
        
        #squared_diffs = [pow(0 if abs(act - pred) < self.error_ok else (act - pred), 2) for act, pred in zip(inputs, targets)]
        # Calculate the mean of the squared differences
        stepmse = sum(squared_diffs) / len(inputs)
        return stepmse


class CustomLossFilter(nn.Module):
        def __init__(self):
            super(CustomLossFilter, self).__init__()

        def forward(self, inputs, targets):
            size = len(inputs)
            if len(targets) != size:
                raise ValueError("The length of actual values and predicted values must be the same")
            # Calculate the squared differences (yi - ybar)^2
            squared_diffs = [abs((inputs[i][0] - inp[i][4]) if abs(inp[i][4] - inp[i][2]) < err_ok else (inputs[i][0] - targets[i][0])) for i in range(size)]
            # Calculate the mean of the squared differences
            stepmse = sum(squared_diffs) / size
            return stepmse