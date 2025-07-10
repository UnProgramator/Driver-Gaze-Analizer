import marshal
import torch
import numpy as np

import pandas as pd
import ast

from typing import Any, Iterable, List, Tuple



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

def readCSV_gt_evaled_loo_drivface(fpath:str, inputDim:int, iset:int|None = 0, vf:str='Original Pitch',gtf:str='Ground Truth Pitch') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None, torch.Tensor|None]:
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

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})
    _vals = df[vf].values

    _vals_set = [None, None, None, None]
    _gt_set = [None, None, None, None]

    for i in range(4):
        i1,i2 = __pic_idx[i]
        _vals_set[i] = [_vals[j:j+inputDim] for j in range(i1-1, i2-inputDim+1)]
        _gt_set[i] = df[gtf].values[(i1-1)+inputDim-1:i2].tolist()
        
    _vals = []
    _gt = []

    for i in [0,1,2,3]:
        if i == iset:
            continue
        _vals += _vals_set[i]
        _gt += _gt_set[i]
 
    Vals_train = torch.tensor(_vals, dtype=torch.float32)
    GT_train = torch.tensor(_gt, dtype=torch.float32).view(-1, 1)

    Vals_test:torch.Tensor|None = None
    GT_test:torch.Tensor|None = None

    if iset is not None:
        Vals_test = torch.tensor(_vals_set[iset], dtype=torch.float32)
        GT_test = torch.tensor(_gt_set[iset], dtype=torch.float32).view(-1, 1)

    return Vals_train, GT_train, Vals_test, GT_test


def readCSV_pitch_or_yaw(fpath:str, inputDim:int, field_v:str|None='Original Pitch', field_gt:str='Ground Truth Pitch',offset:int|None=None) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go
            field_v: str, name of the vals field
            field_gt: str, name of the gt field
            offset: int|None = the position of the value targeted in the ground truth that is the result of a seqence. 
                               If None, then the last element in the sequence is considered the processed value
        Returns:
            (Vals, GT): (Tensor, Tensor), 
                Vals: the input values
                GT: the ground truth
    '''
    
    assert offset is None or offset <= inputDim, 'offset is too large'
    
    stgt = offset if offset is not None else inputDim-1
    endgt=inputDim-1-stgt

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})

    _vals = df[field_v].values
    _vals1 = [_vals[i:i+inputDim] for i in range(len(_vals)-inputDim+1)] # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    _gt = df[field_gt].values[stgt:len(df[field_gt].values)-endgt]

    Vals = torch.tensor(_vals1, dtype=torch.float32)
    GT = torch.tensor(_gt, dtype=torch.float32).view(-1, 1)

    return Vals, GT


def readCSV_pitch_and_yaw(fpath:str, inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   offset:int|None=None) \
                    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go
            pvf, pgf, yvf, ygf: str = name of the corespoding fileds (f) in the csv for the pitch (p) or yaw (y) values (v) or ground truth (g)
            offset: int|None = the position of the value targeted in the ground truth that is the result of a seqence. 
                               If None, then the last element in the sequence is considered the processed value

        Returns:
            (pitch, pGT, yaw, yGT): (Tensor, Tensor, Tensor, Tensor) 
    '''
    assert offset is None or offset <= inputDim, 'offset is too large'
    
    stgt = offset if offset is not None else inputDim-1
    endgt=inputDim-1-stgt

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})

    p_vals = df[pvf].values
    p_vals_tup = [p_vals[i:i+inputDim] for i in range(len(p_vals)-inputDim+1)] # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    p_gt = df[pgf].values[stgt:len( df[pgf].values)-endgt]

    y_vals = df[yvf].values
    y_vals_tup = [y_vals[i:i+inputDim] for i in range(len(y_vals)-inputDim+1)] # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    y_gt = df[ygf].values[stgt:len( df[ygf].values)-endgt]

    pVals = torch.tensor(p_vals_tup, dtype=torch.float32)
    pGT = torch.tensor(p_gt, dtype=torch.float32).view(-1, 1)

    yVals = torch.tensor(y_vals_tup, dtype=torch.float32)
    yGT = torch.tensor(y_gt, dtype=torch.float32).view(-1, 1)

    return pVals, pGT, yVals, yGT

def readCSV_pitch_and_yaw_many_files(fpaths:Iterable[str], inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   offset:int|None=None) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpaths: iterable[str], file paths to the CSV format files
            inputDims: int, number of input values processed in one go
            pvf, pgf, yvf, ygf: str = name of the corespoding fileds (f) in the csv for the pitch (p) or yaw (y) values (v) or ground truth (g)

        Returns:
            (pitch, pGT, yaw, yGT): (Tensor, Tensor, Tensor, Tensor) 
            where the results are the tensors for each file concatenated in one
    '''
    p:list[torch.Tensor] = []
    pgt:list[torch.Tensor] = []
    y:list[torch.Tensor] = []
    ygt:list[torch.Tensor] = []
    for fpath in fpaths:
        p1, pgt1, y1, ygt1 = readCSV_pitch_and_yaw(fpath, inputDim, pvf, pgf, yvf, ygf, offset=offset)
        p.append(p1)
        pgt.append(pgt1)
        y.append(y1)
        ygt.append(ygt1)

    po = torch.cat(p)
    pgto= torch.cat(pgt)
    yo = torch.cat(y)
    ygto= torch.cat(ygt)

    return po, pgto, yo, ygto


def readCSV_pitch_and_yaw_together(fpath:str, inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   offset:int|None=None) \
                    -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go
            pvf, pgf, yvf, ygf: str = name of the corespoding fileds (f) in the csv for the pitch (p) or yaw (y) values (v) or ground truth (g)
            offset: int|None = the position of the value targeted in the ground truth that is the result of a seqence. 
                               If None, then the last element in the sequence is considered the processed value

        Returns:
            (pitch, pGT, yaw, yGT): (Tensor, Tensor, Tensor, Tensor) 
    '''
    assert offset is None or offset <= inputDim, 'offset is too large'
    
    stgt = offset if offset is not None else inputDim-1
    endgt=inputDim-1-stgt

    df = pd.read_csv(fpath,dtype=np.float32) # load pitch training data (csv {Index, OPWnd, GT})

    p_vals:list[np.float32] = list(df[pvf].values)
    y_vals:list[np.float32] = list(df[yvf].values)
    vals_tup = np.array([p_vals[i:i+inputDim] + y_vals[i:i+inputDim] for i in range(len(p_vals)-inputDim+1)],dtype=np.float32) # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    
    
    gt = [[x,y] for x,y in zip(df[pgf].values,df[ygf].values)]
    gt=np.array(gt[stgt:len(gt)-endgt],dtype=np.float32)

    Vals = torch.tensor(vals_tup, dtype=torch.float32)
    GT = torch.tensor(gt, dtype=torch.float32)

    return Vals, GT

def readCSV_pitch_and_yaw_together_many_files(fpaths:Iterable[str], inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   offset:int|None=None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpaths: iterable[str], file paths to the CSV format files
            inputDims: int, number of input values processed in one go
            pvf, pgf, yvf, ygf: str = name of the corespoding fileds (f) in the csv for the pitch (p) or yaw (y) values (v) or ground truth (g)

        Returns:
            (pitch, pGT, yaw, yGT): (Tensor, Tensor, Tensor, Tensor) 
            where the results are the tensors for each file concatenated in one
    '''
    p:list[torch.Tensor] = []
    pgt:list[torch.Tensor] = []

    for fpath in fpaths:
        v, gt = readCSV_pitch_and_yaw_together(fpath, inputDim, pvf, pgf, yvf, ygf, offset=offset)
        p.append(v)
        pgt.append(gt)

    po = torch.cat(p)
    pgto= torch.cat(pgt)

    return po, pgto


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