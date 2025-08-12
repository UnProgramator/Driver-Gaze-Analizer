from dataclasses import dataclass
import marshal
#from warnings import deprecated
import torch
import numpy as np

import pandas as pd
import ast

from typing import IO, Any, Final, Iterable, List, Tuple



# model loader

def saveModel(model:torch.nn.Module, fileName:str, epoch_or_msg:str|int, logFile:IO[str]|None=None):
    print(f"saving model state in file {fileName} for training epoch {epoch_or_msg}", file=logFile)
    torch.save(model,fileName)
    print('model saved succesfully',file=logFile)

def loadModel(fileName:str, logFile:IO[str]|None=None) -> torch.nn.Module:
    print(f'loaging model structure and weights from file {fileName}', file=logFile)
    model:torch.nn.Module = torch.load(fileName, weights_only=False)
    print('Model loaded succesfully', file=logFile)
    return model.eval()

#file loaders

@dataclass
class FieldNames:
    pVals:Final[str]='pVals'
    pGT:Final[str]='pGT'
    yVals:Final[str]='yVals'
    yGT:Final[str]='yGT'
    Vals:Final[str]='Vals'
    GT:Final[str]='GT'
    fErr:Final[str]='fErr'


# 179, 170, 167, 90 total 606, in csv = 605 ???
# 178, 170, 167, 90
__pic_idx = {0:(1,178), 1:(179,348), 2:(349,515), 3:(516,605)}

#@deprecated
def readCSV_gt_evaled_loo_drivface(fpath:str, inputDim:int=5, offset:int|None=None, iset:int|None = None, vf:str='Original Pitch',gtf:str='Ground Truth Pitch') \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None, torch.Tensor|None]:
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

    stgt = offset if offset is not None else inputDim-1
    endgt=inputDim-1-stgt

    # p_vals_tup = [p_vals[i:i+inputDim] for i in range(len(p_vals)-inputDim+1)] # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    # p_gt = df[pgf].values[stgt:len( df[pgf].values)-endgt]
    def __getRange(vals:Iterable,gts:Iterable):
        val = [list(vals[i:i+inputDim]) for i in range(len(vals)-inputDim+1)]
        gt  = list(gts[stgt:len(gts)-endgt])
        return val,gt

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})
    _vals = df[vf].values
    _gt   = df[gtf].values

    _vals_set = [None, None, None, None]
    _gt_set = [None, None, None, None]

    for i in range(4):
        i1,i2 = __pic_idx[i]
        _vals_set[i], _gt_set[i] =__getRange(_vals[i1:i2+1],_gt[i1:i2+1])
        
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

def drivface_readPitch(fpath:str, inputDim:int=5, offset:int|None=None, iset:int|None = None): return readCSV_gt_evaled_loo_drivface(fpath,inputDim,offset=offset,iset=iset,vf='Original Pitch',gtf='Ground Truth Pitch')
def drivface_readYaw  (fpath:str, inputDim:int=5, offset:int|None=None, iset:int|None = None): return readCSV_gt_evaled_loo_drivface(fpath,inputDim,offset=offset,iset=iset,vf='Original Yaw',  gtf='Ground Truth Yaw')

def readCSV_drivface_sep(fpath:str, inputDim:int=5, offset:int|None=None, 
                         pvf:str='Original Pitch',
                         pgtf:str='Ground Truth Pitch', 
                         yvf='Original Yaw',  
                         ygtf='Ground Truth Yaw',
                         fef='Frame Err',
                         bCat:bool=False) \
            -> dict[str, torch.Tensor]|dict[str,list[torch.Tensor]]:
    '''
        Return tensors with the input values and ground truth

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go
            pvf, pgf, yvf, ygf: str = name of the corespoding fileds (f) in the csv for the pitch (p) or yaw (y) values (v) or ground truth (g)
            fef:str = name of the field savingwheter in frame are present interferences (default: "Frame Err")
            offset: int|None = the position of the value targeted in the ground truth that is the result of a seqence. 
                               If None, then the last element in the sequence is considered the processed value

        Returns:
            (Vals, GT, Test, GTTest): (Tensor, Tensor), 
                Vals: the training values
                GT: the ground truth
                Test: test values
                GTTest: Ground truth pentru test
    '''

    stgt = offset if offset is not None else inputDim-1
    endgt=inputDim-1-stgt

    # p_vals_tup = [p_vals[i:i+inputDim] for i in range(len(p_vals)-inputDim+1)] # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    # p_gt = df[pgf].values[stgt:len( df[pgf].values)-endgt]
    def __getRangeVals(vals:Iterable):
        return torch.tensor(np.array([vals[i:i+inputDim] for i in range(len(vals)-inputDim+1)],dtype=np.float32),dtype=torch.float32)
    def __getRangeGtEr(fer:Iterable):
        return torch.tensor(fer[stgt:len(fer)-endgt],dtype=torch.float32)

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})
    _pvals = df[pvf].tolist()
    _pgt   = df[pgtf].tolist()
    _yvals = df[yvf].tolist()
    _ygt   = df[ygtf].tolist()
    _fer   = df[fef].tolist()

    _pvals_set = [None, None, None, None]
    _pgt_set = [None, None, None, None]
    _yvals_set = [None, None, None, None]
    _ygt_set = [None, None, None, None]
    _fer_set = [None, None, None, None]

    for i in range(4):
        i1,i2 = __pic_idx[i]
        _pvals_set[i]= __getRangeVals(_pvals[i1:i2+1])
        _pgt_set[i]  = __getRangeGtEr(_pgt[i1:i2+1]).view(-1,1)
        _yvals_set[i]= __getRangeVals(_yvals[i1:i2+1])
        _ygt_set[i]  = __getRangeGtEr(_ygt[i1:i2+1]).view(-1,1)
        _fer_set[i]  = __getRangeGtEr(_fer[i1:i2+1])

    if bCat:
        po = torch.cat(_pvals_set)
        pgto= torch.cat(_pgt_set)
        yo = torch.cat(_yvals_set)
        ygto= torch.cat(_ygt_set)
        fErro= torch.cat(_fer_set)
        return {FieldNames.pVals:po, FieldNames.pGT:pgto, FieldNames.yVals:yo, FieldNames.yGT:ygto, FieldNames.fErr:fErro}

    return {FieldNames.pVals:_pvals_set, FieldNames.pGT:_pgt_set, FieldNames.yVals:_yvals_set, FieldNames.yGT:_ygt_set, FieldNames.fErr:_fer_set}


def readCSV_pitch_and_yaw(fpath:str, inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   fef:str='Frame Err',
                                                   offset:int|None=None) \
                    -> dict[str,torch.Tensor]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpath: str, file path to the CSV format file
            inputDims: int, number of input values processed in one go
            pvf, pgf, yvf, ygf: str = name of the corespoding fileds (f) in the csv for the pitch (p) or yaw (y) values (v) or ground truth (g)
            fef:str = name of the field savingwheter in frame are present interferences (default: "Frame Err")
            offset: int|None = the position of the value targeted in the ground truth that is the result of a seqence. 
                               If None, then the last element in the sequence is considered the processed value

        Returns:
            (pitch, pGT, yaw, yGT): (Tensor, Tensor, Tensor, Tensor) 
    '''
    assert offset is None or offset < inputDim, 'offset is too large'
    
    retVal = dict()

    stgt = offset if offset is not None else inputDim-1
    endgt=inputDim-1-stgt

    df = pd.read_csv(fpath) # load pitch training data (csv {Index, OPWnd, GT})

    p_vals = df[pvf].values
    p_vals_tup = np.array([p_vals[i:i+inputDim] for i in range(len(p_vals)-inputDim+1)], dtype=np.float32) # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    p_gt = df[pgf].values[stgt:len( df[pgf].values)-endgt]

    y_vals = df[yvf].values
    y_vals_tup = np.array([y_vals[i:i+inputDim] for i in range(len(y_vals)-inputDim+1)], dtype=np.float32) # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    y_gt = df[ygf].values[stgt:len( df[ygf].values)-endgt]

    pVals = torch.tensor(p_vals_tup, dtype=torch.float32)
    pGT = torch.tensor(p_gt, dtype=torch.float32).view(-1, 1)

    yVals = torch.tensor(y_vals_tup, dtype=torch.float32)
    yGT = torch.tensor(y_gt, dtype=torch.float32).view(-1, 1)

    efe = y_gt = df[fef].values[stgt:len( df[ygf].values)-endgt]
    fErr = torch.tensor(efe, dtype=torch.int8).view(-1, 1)

    retVal[FieldNames.pVals] = pVals
    retVal[FieldNames.pGT] = pGT
    retVal[FieldNames.yVals] = yVals
    retVal[FieldNames.yGT] = yGT
    retVal[FieldNames.fErr] = fErr

    return retVal



def readCSV_pitch_and_yaw_many_files(fpaths:Iterable[str], inputDim:int, 
                                                   pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   fef:str='Frame Err',
                                                   offset:int|None=None,
                                                   bCat:bool=True) \
        -> dict[str,torch.Tensor]|dict[str,list[torch.Tensor]]:
    '''
        Returns tensors with the input values and the ground turth, for either training or validation

        Args:
            fpaths: iterable[str], file paths to the CSV format files
            inputDims: int, number of input values processed in one go
            pvf, pgf, yvf, ygf: str = name of the corespoding fileds (f) in the csv for the pitch (p) or yaw (y) values (v) or ground truth (g)
            fef:str = name of the field savingwheter in frame are present interferences (default: "Frame Err")
            offset: int|None = the position of the value targeted in the ground truth that is the result of a seqence. 
                               If None, then the last element in the sequence is considered the processed value
            bCat: bool = whether to return the sensors concatenated or as list

        Returns:
            (pitch, pGT, yaw, yGT): (Tensor, Tensor, Tensor, Tensor) 
            where the results are the tensors for each file concatenated in one
    '''
    p:list[torch.Tensor] = []
    pgt:list[torch.Tensor] = []
    y:list[torch.Tensor] = []
    ygt:list[torch.Tensor] = []
    fErr:list[torch.Tensor] = []
    for fpath in fpaths:
        retVal = readCSV_pitch_and_yaw(fpath, inputDim, pvf, pgf, yvf, ygf, offset=offset)
        p.append(retVal[FieldNames.pVals])
        pgt.append(retVal[FieldNames.pGT])
        y.append(retVal[FieldNames.yVals])
        ygt.append(retVal[FieldNames.yGT])
        fErr.append(retVal[FieldNames.fErr])

    if bCat:
        po = torch.cat(p)
        pgto= torch.cat(pgt)
        yo = torch.cat(y)
        ygto= torch.cat(ygt)
        fErro= torch.cat(fErr)
        return {FieldNames.pVals:po, FieldNames.pGT:pgto, FieldNames.yVals:yo, FieldNames.yGT:ygto, FieldNames.fErr:fErro}

    return {FieldNames.pVals:p, FieldNames.pGT:pgt, FieldNames.yVals:y, FieldNames.yGT:ygt, FieldNames.fErr:fErr}


def readCSV_pitch_and_yaw_together(fpath:str, inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   fef:str='Frame Err',
                                                   offset:int|None=None) \
                    -> dict[str,torch.Tensor]:
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

    p_vals:list[np.float32] = df[pvf].tolist()
    y_vals:list[np.float32] = df[yvf].tolist()
    vals_tup = [p_vals[i:i+inputDim] + y_vals[i:i+inputDim] for i in range(len(p_vals)-inputDim+1)] # cause _vals[i:i+inputDim] takes the elems with the index in [i, i+inputDim), why +1?
    
    gt = [[x,y] for x,y in zip(df[pgf].values,df[ygf].values)]
    gt=gt[stgt:len(gt)-endgt]

    fer = df[fef].tolist()

    Vals = torch.tensor(vals_tup, dtype=torch.float32)
    GT = torch.tensor(gt, dtype=torch.float32)
    FErr = torch.tensor(fer[stgt:len(fer)-endgt], dtype=torch.uint8)#.view(-1,1)
    #FErr = torch.cat((FErr,FErr),dim=1)

    return {FieldNames.Vals:Vals, FieldNames.GT:GT, FieldNames.fErr:FErr}

def readCSV_pitch_and_yaw_together_many_files(fpaths:Iterable[str], inputDim:int, pvf:str='Original Pitch', 
                                                   pgf:str='Ground Truth Pitch', 
                                                   yvf:str='Original Yaw', 
                                                   ygf:str='Ground Truth Yaw',
                                                   offset:int|None=None,
                                                   bCat:bool=True
                                                   ) \
        -> dict[str,torch.Tensor]|dict[str,list[torch.Tensor]]:
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
    fer:list[torch.Tensor] = []

    for fpath in fpaths:
        dd = readCSV_pitch_and_yaw_together(fpath, inputDim, pvf, pgf, yvf, ygf, offset=offset)
        v, gt,fe = dd[FieldNames.Vals], dd[FieldNames.GT], dd[FieldNames.fErr]
        p.append(v)
        pgt.append(gt)
        fer.append(fe)

    if bCat:
        po   = torch.cat(p)
        pgto = torch.cat(pgt)
        fero = torch.cat(fer)
        return {FieldNames.Vals:po, FieldNames.GT:pgto, FieldNames.fErr:fero}

    return {FieldNames.Vals:p, FieldNames.GT:pgt, FieldNames.fErr:fer}

#need to be tested
def extractframesWithProblems_fromDic( src : dict[str,torch.Tensor]|dict[str,list[torch.tensor]],
                                       padding: int,
                                       fields: list[str]|None=None
                                     ) -> dict[str,list[torch.Tensor]]:

    if fields is None: fields = list(src.keys())
    assert FieldNames.fErr in src, 'field \'fErr\' was not presnt in the source'

    errT:list[torch.Tensor] = src[FieldNames.fErr] if isinstance(src[FieldNames.fErr],list) else [src[FieldNames.fErr]]

    other:list[list[torch.Tensor]] = [
            (src[fl] if isinstance(src[fl],list) else [src[fl]]) for fl in fields
        ]

    def genList(erVect:torch.Tensor)->list[int]:
        retVal = []
        for i in range(len(erVect)):
            if i-1 not in retVal: retVal.append(i-1)
            if i not in retVal: retVal.append(i)
            if i+1 not in retVal: retVal.append(i+1)
        return retVal

    def carve(otherVect:torch.Tensor, idxs:list[int]):
        tp = otherVect.dtype
        temp = [otherVect[i] for i in idxs]
        return torch.tensor(temp, dtype=tp)

    def process(erTen:torch.Tensor, otherTens:list[torch.Tensor]):
        erIdxs=genList(erTen)
        retVal = []
        for t in otherTens:
            r = carve(t,erIdxs)
            retVal.append(r)
        return retVal

    res = []

    for i in len(errT):
        er:torch.Tensor = errT[i]
        ot:list[torch.Tensor] = [o[i] for o in other]
        res.append(process(er,ot))

    retVal:dict[str,list[torch.Tensor]]={}
    i=0
    for k in fields:
        retVal[k] = [rs[i] for rs in res]
        i+=1

    return retVal