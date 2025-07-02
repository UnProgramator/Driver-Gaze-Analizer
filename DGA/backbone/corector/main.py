from typing import IO
from losfucts import CustomLoss, CustomLoss_v
from CorectionUtilities import *
from corectors import train, validate, loadModel, saveModel


from datetime import datetime
from SimpleNN import SimpleNN


import torch

# infile1:str = "C:/Users/apesc/Downloads/pitch_train_data.csv"

resultsPath='D:/DCIM/results/'

plotfile:str  = resultsPath+"plots/{}.png"
logFolder = resultsPath+'logs/'
logfile:str   = logFolder+'results.log'
modelFile:str = resultsPath+"models/"

dataPath=resultsPath+'data/'

infile:str = dataPath+"Err_gaze.csv"

print_train = False

datasets:list[str] = [dataPath+'err_b1.csv',
                      dataPath+'err_b2.csv',
                      dataPath+'err_b3.csv',
                      dataPath+'err_b4.csv']

epocs = 8000  # training epocs
err_ok = 0.05 # estimated pitch - maximum admited error



def steps(vals:torch.Tensor, gt:torch.Tensor, test_vals:torch.Tensor, test_gt:torch.Tensor, log:IO[str],name:str):
    #model:nn.Module = SimpleNN.default() # 1,2,3

    model:torch.nn.Module = SimpleNN([5, 15]) # 5, 6

    #model:nn.Module = SimpleNN([5, 5]) # 7, 8

    #model= SimpleNN([5, 120, 10]) # 9

    #model:nn.Module = SimpleNN([5, 30]) # 11

    #training phase:

    #model = train(10000, model, inp, gt)    
    model = train(1000, 
                  model, 
                  vals, 
                  gt, 
                  CustomLoss_v(err_ok),logFile=log,saveSteps=[500,1000],modelSaveFileTemplate=resultsPath+'temp/{}.model')    
    #model = train(20000, model ,inp, gt)
    #model = train(16000, model ,inp, gt)

    #model = _train(epocs, model, vals, losFn, logFile)

    print(model.eval())
 

    print("evaluate on the training data")

    validate(model, vals, gt, err_ok, plotSavefile=plotfile.format(name+'_for_train_TEST'))


    print("pe datele de validare")

    validate(model, test_vals, test_gt, err_ok, plotSavefile=plotfile.format(name+'_for_valid_TEST'))

    
def test1(drv:int = 3):
    f = open('D:/DCIM/test.log', 'w+')
    #inp,gt = readCSV_gt_evaled(infile1)

    #inp, gt, intt, gttt = readCSV_gt_evaled_loo_drivface(infile, 5, drv)
    # assert (intt is not None and gttt is not None)

    pv, pgt, yv, ygt = readCSV_pitch_and_yaw_many_files(datasets,5)

    tpv, tpgt, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None)
    tyv, tygt, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None, 'Original Yaw', 'Ground Truth Yaw')

    # print(inp)
    # print(gt)
    # print(intt)
    # print(gttt)

    print('Info for pitch')
    steps(pv, pgt, tpv, tpgt,log=f,name='pitch')

    print('Info for yaw')
    steps(yv, ygt, tyv, tygt,log=f,name='yaw')


def test2():
    with open('D:/DCIM/test.log', 'w+') as lf:
        model:torch.nn.Module = SimpleNN([5, 15])
        pv, pgt, yv, ygt = readCSV_pitch_and_yaw_many_files(datasets,5)
        print('\n\n')
        model = train(140, 
                      model, 
                      pv, 
                      pgt, 
                      CustomLoss(err_ok),logFile=lf) 
        print(model.eval(),file=lf)
        saveModel(model, "D:/DCIM/models/save_test.model",'final',logFile=lf)
        nm = loadModel("D:/DCIM/models/save_test.model",lf)
        print(nm,file=lf)

models:dict[int,tuple[type[torch.nn.Module],list[int],torch.nn.Module|None]] 

check_points:list[int]
#checkpoint_temp_name = 'D:/DCIM/models/tests/'
#log_path = 'D:/DCIM/models/logs.log'

exp_msg = 'experiment {exp_name}, starting at {tm}'

def exp1(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss()):
    flog = open(logfile, 'a+')
    pv, pgt, yv, ygt = readCSV_pitch_and_yaw_many_files(datasets,5)

    modtp,layers,losfun = models[exp_no]
    model_p=modtp(layers,losfun)
    print(exp_msg.format(exp_name='exp1'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+'-exp1-yaw-epoch{}.model')
    model_y=modtp(layers,losfun)
    model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+'-exp1-pitch-epoch{}.model')

def exp2(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss()):
    flog = open(logfile, 'a+')
    pv, pgt, yv, ygt = readCSV_pitch_and_yaw_many_files(datasets[1:],5)

    modtp,layers,losfun = models[exp_no]
    model_p=modtp(layers,losfun)
    print(exp_msg.format(exp_name='exp2'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+'-exp2-yaw-epoch{}.model')
    model_y=modtp(layers,losfun)
    model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+'-exp2-pitch-epoch{}.model')

def exp3(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss()):
    flog = open(logfile, 'a+')
    pv, pgt, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None)
    yv, ygt, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None, 'Original Yaw', 'Ground Truth Yaw')

    modtp,layers,losfun = models[exp_no]
    model_p=modtp(layers,losfun)
    print(exp_msg.format(exp_name='exp3'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+'-exp3-yaw-epoch{}.model')
    model_y=modtp(layers,losfun)
    model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+'-exp3-pitch-epoch{}.model')


import ntpath
import os
import time


def validateAllModels(folder:str|None=None, files:list[str]|None=None, log:IO[str]|None=None, results:IO[str]|None=None, plot_on:bool=False, resCSV:IO[str]|None=None):
    pv4, pgt4, yv4, ygt4 = readCSV_pitch_and_yaw_many_files(datasets,5)
    pv3, pgt3, yv3, ygt3 = readCSV_pitch_and_yaw_many_files(datasets[1:],5)
    pvd, pgtd, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None)
    yvd, ygtd, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None, 'Original Yaw', 'Ground Truth Yaw')

    exp_code:str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csvFile:IO[str]

    def __validate_model(filePath:str, valiName:str, vals:torch.Tensor, gt:torch.Tensor):
        print(f'Testing model {filePath} with dataset {valiName}',file=results)
        model=loadModel(filePath,log)
        filename=ntpath.basename(filePath) +'__' + valiName
        iacc,ialoss,tacc,taloss = validate(model,vals,gt,logFile=results,plotSavefile=(plotfile.format(filename) if plot_on else None) )
        print(exp_code,filename,valiName,iacc,ialoss,tacc,taloss,sep=',',file=csvFile)
        print('--------------------------------------------------------------------\n', file=results)

    def __validate_wrap(filePath:str):
        if 'yaw' in filePath:
            __validate_model(filePath,'OURS_4',yv4,ygt4)
            __validate_model(filePath,'OURS_3',yv3,ygt3)
            __validate_model(filePath,'Drivface',yvd,ygtd)
        elif 'pitch' in filePath:
            __validate_model(filePath,'OURS_4',pv4,pgt4)
            __validate_model(filePath,'OURS_3',pv3,pgt3)
            __validate_model(filePath,'Drivface',pvd,pgtd)
        else: raise Exception()
        print('\n======================================================================\n\n', file=results)

    d1 = time.time()
    tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'~~~~~~~~~~~~~~~Validation at {tm}~~~~~~~~~~~~~~~n\n', file=results)
    

    if files is None:
        if folder is not None:
            files = [folder+f for f in os.listdir(folder)]
        else:
            raise Exception

    csvFile = open(logFolder+'mean_error_and_loss.csv','w+')
    #print('Experiemnt Code,Model Name,Validation Set,Initial Accracy,Initial Average Average Loss,Test Average Accracy,Test Avergae Loss',file=csvFile)
    print('expcode,model,validset,iacc,ialoss,tacc,taloss',file=csvFile)

    for f in files:
        print(f'Validate mode {f}', file=results)
        __validate_wrap(f)

    print(f'Total time: {time.time()-d1} sec', file=results)
    print(f'~~~~~~~~~~~~~~~End of validation at {tm}~~~~~~~~~~~~~~~n\n', file=results)

def defaultValidation():
    with open(logFolder+'validationsLogs.log','a+') as lf, open(logFolder+'validationResults.log','a+') as resf:
        validateAllModels(folder=modelFile,log=lf,results=resf)


check_points= [2000, 5000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000]
models= {
        0:(SimpleNN,[5, 15],None),
        1:(SimpleNN,[5, 25],None),
        2:(SimpleNN,[5, 25, 15],None),
        3:(SimpleNN,[5, 15, 25],None),
        4:(SimpleNN,[5, 15, 5],None),
        5:(SimpleNN,[5, 15, 1],None),
        6:(SimpleNN,[5, 15, 25, 5],None)
}

if __name__ == '__main__':

    # exp1(exp_no=1, epocs=30_000)
    # exp1(exp_no=1, epocs=30_000,losFn=CustomLoss_v(err_ok))
    # exp2(exp_no=1, epocs=30_000)
    # exp3(exp_no=1, epocs=30_000)
    # exp3(exp_no=2, epocs=30_000)

    # print('start')
    # exp1(exp_no=0, epocs=40_000)
    # exp1(exp_no=2, epocs=40_000)
    # exp2(exp_no=0, epocs=40_000)
    # exp2(exp_no=2, epocs=40_000)
    # exp1(exp_no=4, epocs=40_000)
    # exp2(exp_no=4, epocs=40_000)
    # exp1(exp_no=6, epocs=40_000)
    # exp2(exp_no=6, epocs=40_000)
    
    defaultValidation()