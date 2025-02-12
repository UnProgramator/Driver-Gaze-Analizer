import marshal
import torch
import numpy as np

import pandas as pd
import ast

from typing import Any, List, Tuple



def readCSV_gt_evaled(fpath:str) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Return tensors with the input values and ground truth

        Args:
            fpath: str, file path to the CSV format file

        Returns:
            (Vals, GT): (Tensor, Tensor), 
                Vals: the training values
                GT: the ground truth
    '''

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})

    df['OPWnd'] = df['OPWnd'].apply(ast.literal_eval)

    Vals = torch.tensor(df['OPWnd'].tolist(), dtype=torch.float32)
    GT = torch.tensor(df['GT'].values, dtype=torch.float32).view(-1, 1)

    return Vals, GT

# 179, 170, 167, 90 total 606, in csv = 605 ???
# 178, 170, 167, 90
__pic_idx = {0:(1,178), 1:(179,348), 2:(349,515), 3:(516,605)}

def readCSV_gt_evaled_loo_drivface(fpath:str, inputDim:int, iset:int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
        Return tensors with the input values and ground truth

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go
            iset: int, index of the set let out of training

        Returns:
            (Vals, GT, Test, GTTest): (Tensor, Tensor), 
                Vals: the training values
                GT: the ground truth
                Test: test values
                GTTest: Ground truth pentru test
    '''

    i1,i2 = __pic_idx[iset]
    i1 = i1-1

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})
    _vals = df['Original Pitch'].values

    _vals_set = [None, None, None, None]
    _gt_set = [None, None, None, None]

    for i in range(4):
        i1,i2 = __pic_idx[i]
        _vals_set[i] = [_vals[j:j+inputDim] for j in range(i1-1, i2-inputDim)]
        _gt_set[i] = df['Ground Truth Pitch'].values[i1+inputDim-1:i2].tolist()
        
    _vals = []
    _gt = []

    for i in [0,1,2,3]:
        if i == iset:
            continue
        _vals += _vals_set[i]
        _gt += _gt_set[i]
 
    Vals_train = torch.tensor(_vals, dtype=torch.float32)
    GT_train = torch.tensor(_gt, dtype=torch.float32).view(-1, 1)

    Vals_test = torch.tensor(_vals_set[iset], dtype=torch.float32)
    GT_test = torch.tensor(_gt_set[iset], dtype=torch.float32).view(-1, 1)

    return Vals_train, GT_train, Vals_test, GT_test


def readCSV_gt(fpath:str, inputDim:int, field_v:str='Original Pitch', field_gt:str='Ground Truth Pitch') -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go

        Returns:
            (Vals, GT): (Tensor, Tensor), 
                Vals: the input values
                GT: the ground truth
    '''

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})

    _vals = df['Original Pitch'].values
    _vals = [_vals[i:i+inputDim] for i in range(len(_vals)-inputDim)] # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim)
    _gt = df['Ground Truth Pitch'].values[inputDim-1:]

    Vals = torch.tensor(_vals, dtype=torch.float32)
    GT = torch.tensor(_gt, dtype=torch.float32).view(-1, 1)

    return Vals, GT

def readCSV(fpath:str, inputDim:int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Returns a tensor with the input values and the ground turth, for either training or validation

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go

        Returns:
            Vals: Tensor, the input values
    '''
    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})

    _vals = df['Original Pitch'].values
    _vals = [_vals[i:i+inputDim] for i in range(len(_vals)-inputDim)]

    Vals = torch.tensor(_vals, dtype=torch.float32)

    return Vals




def tensorFromArray(ar):
    return torch.tensor(ar, dtype=torch.float32)

def inputFromArrayForTraining(pred:List[Tuple[int,int]] ,gt:List[Tuple[int,int]]) -> torch.Tensor:
    dim = len(pred)

    retval = []

    for i in range(4,dim):
        batch = pred[i-4,i]
        retval.append([batch, gt[i]])

    return retval

def inputFromArrayAtRunning(ar:List[Tuple[int,int]]) -> torch.Tensor:
    dim = len(ar)

    retval = []

    for i in range(4,dim):
        batch = ar[i-4,i]
        retval.append(batch)

    return retval

def tensorFromInput(ar:List[Tuple[int,int]]) -> List:
    return torch.tensor(ar)

@DeprecationWarning
def arrayFromCSV(file:str) -> Tuple[ List[Tuple[int,int]], List[Tuple[int,int]], List[int] ]:
    '''
    Deprecated
    '''

    import csv
    retval = ([],[], [])

    #read csv as vector
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\'')
        for row in spamreader:
            #form the array
            fr = row['Frame']
            p_pitch  = row['Original Pitch']
            p_yaw    = row['Original Yaw']
            gt_pitch = row['Ground Truth Pitch']
            gt_yaw   = row['Ground Truth Yaw']

            retval[0].append((p_pitch, p_yaw))
            retval[1].append((gt_pitch, gt_yaw))
            retval[2].append(fr)

    return retval


def serialize(ar:Any, file:str):
    with open(file, 'w') as fd:
        fd.write(marshal.dumps(ar))

def deserialize(file:str):
    retval = None
    with open(file, 'r') as fd:
        retval = marshal.loads(fd.read())

def inputToText(ar:List[List], file:str) -> None:
    with open(file, 'w') as fd:
        if row[0][0] is List:
            fd.write('gt+p\n')
            for row in ar:
                fd.write(str(row[1]) + '|' + str(row) + '\n')
        else:
            fd.write('p\n')
            for row in ar:
                fd.write(str(row) + '\n')

def inputFromText(file:str) -> List:
    vec = []
    with open(file, 'r') as fd:
        tip = fd.readline()
        if tip == 'gt+p\n':
            for row in fd.readlines():
                pred,vec = row.split('|')
                el = [[],float(pred)]
                pass
        elif tip == 'p\n':
            for row in fd.readlines():
                pass
    return vec