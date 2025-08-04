from functools import reduce
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import IO

import pandas as pd

def saveModel(model:torch.nn.Module, fileName:str, epoch_or_msg:str|int, logFile:IO[str]|None=None):
    print(f"saving model state in file {fileName} for training epoch {epoch_or_msg}", file=logFile)
    torch.save(model,fileName)
    print('model saved succesfully',file=logFile)

def loadModel(fileName:str, logFile:IO[str]|None=None) -> torch.nn.Module:
    print(f'loaging model structure and weights from file {fileName}', file=logFile)
    model:torch.nn.Module = torch.load(fileName, weights_only=False)
    print('Model loaded succesfully', file=logFile)
    return model.eval()

def _train(epocs:int, 
           model:torch.nn.Module, 
           data:torch.Tensor, 
           gt:torch.Tensor, 
           lossFn:torch.nn.Module,
           logFile:IO[str]|None=None,
           saveSteps:list[int]|None=None,
           modelSaveFileTemplate:str|None=None
           ) -> torch.nn.Module:
    # NN loss and optimizer
    #criterion = nn.MSELoss()  # Mean Squared Error Loss
    #criterion = CustomLoss(err_ok)

    #verify that save path is not None when save steps is not None
    assert saveSteps is None or modelSaveFileTemplate is not None ,\
           'modelSaveFileTemplate not given when saveSteps given'

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    step = epocs//70
    string='|'+' '*70+'|'
    # NN training
    for epoch in range(1,epocs+1):
        if epoch%step == 0:
            string = '|' + '='*(epoch//step) + ' '*(70-epoch//step) + '|'
        perce=f'{epoch/epocs*100:06.2f}%'
        print(string[0:34], perce, string[35:] ,end='\r',sep=None)          
        optimizer.zero_grad()         # gradients reset to 0
        outputs = model(data)      
        loss = lossFn(outputs, gt)  # loss computation
        loss.backward()               # backpropagation
        optimizer.step()              # weights adjustement
        
        if saveSteps is not None and epoch in saveSteps: # do not save at the last step, as it will be saved anyway
            saveModel(model,modelSaveFileTemplate.format(epoch), epoch, logFile)

        if logFile and (epoch + 1) % (epocs/10) == 0:
            print(f"Epoch {epoch+1}/{epocs}, Loss: {loss.item():.4f}", file=logFile) # display loss during training
    print()
    if modelSaveFileTemplate is not None and (saveSteps is None or epocs not in saveSteps):
        saveModel(model,modelSaveFileTemplate.format(epocs), 'finish of training', logFile)

    return model


# def _validate_old(model:torch.nn.Module, invals:torch.Tensor, gtv:torch.Tensor, err_ok:float, logFile:IO[str]|None=None, plot_save_file:str|None=None,show_plot:bool=False)\
#     -> tuple[float,float,float,float]:
#     #results are (pitch,yaw)
#     # Model testing (Problem: only one dataset {X,y}, also for evaluation)
#     model.eval()

#     predictions:torch.Tensor = model(invals)
#     test_loss, correct = 0, 0
#     initial_loss, inital_correct = 0, 0
#     loss_fn = torch.nn.MSELoss(reduction='sum')  # conteaza cu ce am antrenat?
#     size = invals.size()[0]
#     with torch.no_grad():  
#       for i in range(size):
#         Xc:torch.Tensor = invals[i]
#         yc:torch.Tensor = gtv[i]
#         pred = predictions[i]
#         # print(str(type(Xc)) + ' ' + str(type(yc)) + ' ' + str(type(pred)))
#         test_loss += loss_fn(pred, gtv[i]).item()
#         correct += int(abs(pred - gtv[i].item()[0]) <= err_ok)

#         initial_pred = torch.tensor([Xc[-1]], requires_grad=True)

#         initial_loss += loss_fn(initial_pred, gtv[i]).item()
#         inital_correct += int(abs(initial_pred.item() - yc[0]) <= err_ok)

#     tacc=(100*correct/size)
#     taloss=np.sqrt(test_loss)/100
#     iacc=(100*inital_correct/size)
#     ialoss=np.sqrt(initial_loss)/100

#     print(f"Test Error: \n Accuracy(diff < {err_ok:>0.3f}): {tacc}%, Avg loss: {taloss:>8f} \n", file=logFile)
#     print(f"Initial Error: \n Accuracy(diff < {err_ok:>0.3f}): {iacc:>0.1f}%, Avg loss: {ialoss:>8f} \n", file=logFile)

#     #print([r[4] for r in df['OPWnd'].tolist()])
#     if show_plot or plot_save_file:
#         vdf =  pd.DataFrame({'GT':gtv.squeeze().tolist(), 'l2cs_net':[r[-1] for r in invals.squeeze().tolist()],'Err_corr':predictions.squeeze().tolist()})
#         graph = vdf.plot(title='Err correction', figsize=(100,20)) # modify for dinamic dimension
#         if plot_save_file: plt.savefig(plot_save_file)
#         if show_plot: plt.show()

#     return iacc,ialoss,tacc,taloss

def _validate(model:torch.nn.Module, inputVals:torch.Tensor, gtVals:torch.Tensor, initialVals:torch.Tensor, err_ok:float, logFile:IO[str]|None=None, plot_save_file_template:str|None=None, show_plot:bool=False)\
            -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    predictions:torch.Tensor = model(inputVals)
    
    taloss = torch.sqrt((torch.square(predictions-gtVals)).mean(dim=0,keepdim=True)).flatten()
    ialoss = torch.sqrt((torch.square(initialVals-gtVals)).mean(dim=0,keepdim=True)).flatten()

    tacc = (torch.abs(predictions-gtVals) <= err_ok).mean(dim=0,keepdim=True,dtype=torch.float32).flatten()*100
    iacc = (torch.abs(initialVals-gtVals) <= err_ok).mean(dim=0,keepdim=True,dtype=torch.float32).flatten()*100

    if(taloss.shape[0]==1):
        print()

    for i in range(tacc.shape[0]):
        print(f"Test Error,    position{i}: Accuracy(diff < {err_ok:02.2f}): {tacc[i].item():02.2f}%, Avg loss: {taloss[i].item():08.2f}", file=logFile)
        print(f"Initial Error, position{i}: Accuracy(diff < {err_ok:02.2f}): {iacc[i].item():02.2f}%, Avg loss: {ialoss[i].item():08.2f}", file=logFile)

    #print([r[4] for r in df['OPWnd'].tolist()])
    if plot_save_file_template is not None:
        valDict = {'GT':gtVals.squeeze().tolist(), 'l2cs_net':initialVals.squeeze().tolist(),'Err_corr':predictions.squeeze().tolist()}
        vdf =  pd.DataFrame(valDict)
        graph = vdf.plot(title='Err correction', figsize=(100,20)) # modify for dinamic dimension
        if plot_save_file_template: plt.savefig(plot_save_file_template.format('val'))
        if show_plot: plt.show()

        valDict = {'l2cs_net-acc':(initialVals==gtVals).type(dtype=torch.int32).squeeze().tolist(),'proces-acc':(torch.abs(predictions-gtVals)>0.000001).type(dtype=torch.int32).squeeze().tolist()}
        vdf =  pd.DataFrame(valDict)
        graph = vdf.plot(title='Err correction', figsize=(100,20)) # modify for dinamic dimension
        if plot_save_file_template: plt.savefig(plot_save_file_template.format('acc'))
        if show_plot: plt.show()

    return iacc,ialoss,tacc,taloss




def train(epocs:int, 
          model:torch.nn.Module, 
          vals:torch.Tensor, gt:torch.Tensor,  
          losFn:torch.nn.Module|None=None , 
          logFile:IO[str]|None=None,
          saveSteps:list[int]|None=None, modelSaveFileTemplate:str|None=None) -> torch.nn.Module:
    #training

    if losFn == None:
        losFn = torch.nn.MSELoss()

    start_time = time.time()
    model = _train(epocs, model, vals, gt, losFn, 
                   logFile=logFile,
                   saveSteps=saveSteps,
                   modelSaveFileTemplate=modelSaveFileTemplate)
    end_time = time.time()
    
    print(f'training end in {end_time - start_time} seconds')
    if logFile: print(f'training end in {end_time - start_time} seconds',file=logFile)
    
    return model

def validate(model:torch.nn.Module, inputVals:torch.Tensor, gtVals:torch.Tensor, initialVals:torch.Tensor, datasetName:str='dataset not specified', err_ok:float=0.05, logFile:IO[str]|None=None, plotSavefile:str|None=None)\
            -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    print(f'evaluate on the {datasetName} data')
    if logFile: print(f'evaluating the model, using data labeled{datasetName}', file=logFile)
    start_time = time.time()
    r = _validate(model=model, inputVals=inputVals, gtVals=gtVals,initialVals=initialVals, err_ok=err_ok,logFile=logFile,plot_save_file_template=plotSavefile)
    end_time = time.time()
    print(f'validation end in {end_time - start_time} seconds')
    if logFile: print(f'validation end in {end_time - start_time} seconds', file=logFile)
    return r


