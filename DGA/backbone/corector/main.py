from typing import IO
from validation import Validation
from losfucts import CustomLoss, CustomLoss_v, INamedModule
from CorectionUtilities import *
from corectors import train, validate, loadModel, saveModel

from CustomArchNN1 import CustomNN_YArch
from CustomArchNN2 import CustomNN_MArch
from DeciderNN import DeciderNN
from SimpleNN import SimpleNN

from datetime import datetime
from SimpleNN import SimpleNN


import torch

# infile1:str = "C:/Users/apesc/Downloads/pitch_train_data.csv"

resultsPath='D:/DCIM/images_12fps/experiment/'

plotfile:str  = resultsPath+"plots/{}.png"
logFolder = resultsPath+'logs/'
logfile:str   = logFolder+'results.log'
modelFile:str = resultsPath+"models/"

dataPath=resultsPath+'data/'

infile:str = "D:/DCIM/results/data/Err_gaze.csv"

print_train = False

datasets:list[str] = [dataPath+'err_b1.csv',
                      dataPath+'err_b2.csv',
                      dataPath+'err_b3.csv',
                      dataPath+'err_b4.csv',
                      dataPath+'err_b5.csv',
                      dataPath+'err_b6.csv',
                      dataPath+'err_b7.csv',
                      dataPath+'err_b8.csv',
                      dataPath+'err_b9.csv']

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

models:dict[int,tuple[type[torch.nn.Module],list[int]|list[list[int]],list[type[torch.nn.Module]]|type[torch.nn.Module]|None]] 

check_points:list[int]
#checkpoint_temp_name = 'D:/DCIM/models/tests/'
#log_path = 'D:/DCIM/models/logs.log'

exp_msg = 'experiment {exp_name}, starting at {tm}, with seed {sed}'

def exp1(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(),offset:int|None=None, inputDims:int=5):
    flog = open(logfile, 'a+')
    pv, pgt, yv, ygt = readCSV_pitch_and_yaw_many_files(datasets,inputDims,offset=offset)

    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()

    modtp,layers,losfun = models[exp_no]
    layers[0]=inputDims
    model_p=modtp(layers,losfun)
    seed=torch.seed()
    print(exp_msg.format(exp_name='exp1'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp1-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
    model_y=modtp(layers,losfun)
    model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-exp1-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')

def exp2(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(),offset:int|None=None, inputDims:int=5):
    flog = open(logfile, 'a+')
    pv, pgt, yv, ygt = readCSV_pitch_and_yaw_many_files(datasets[1:],5)

    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()
    modtp,layers,losfun = models[exp_no]
    seed=torch.seed()
    model_p=modtp(layers,losfun)
    print(exp_msg.format(exp_name='exp2'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp2-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-exp2-yaw-epoch{}.model')
    model_y=modtp(layers,losfun)
    model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-exp2-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-exp2-pitch-epoch{}.model')

def exp3(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(), offset:int|None=None, inputDims:int=5):
    flog = open(logfile, 'a+')
    pv, pgt, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None)
    yv, ygt, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None, 'Original Yaw', 'Ground Truth Yaw')

    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()
    modtp,layers,losfun = models[exp_no]
    model_p=modtp(layers,losfun)
    seed=torch.seed()
    print(exp_msg.format(exp_name='exp3'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp3-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-exp3-yaw-epoch{}.model')
    model_y=modtp(layers,losfun)
    model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-exp3-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-exp3-pitch-epoch{}.model')

def exp4(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(),offset:int|None=None,inputDims:int=5):
    flog = open(logfile, 'a+')
    pv, pgt = readCSV_pitch_and_yaw_together_many_files(datasets,inputDims,offset=offset)

    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()

    modtp,layers,losfun = models[exp_no]
    layers[0]=2*inputDims
    seed=torch.seed()
    model_p=modtp(layers,losfun)
    print(exp_msg.format(exp_name='exp5'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp4-pitch_and_yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')

def exp5(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(),offset:int|None=None,inputDims:int=5):
    flog = open(logfile, 'a+')
    pv, pgt = readCSV_pitch_and_yaw_together_many_files(datasets,inputDims,offset=offset)

    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()

    modtp,layers,losfun = models[exp_no]
    layers[0]=2*inputDims
    layers[-1]=1
    seed=torch.seed()
    model_p=modtp(layers,losfun)
    print(exp_msg.format(exp_name='exp5'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
    model_p = train(epocs, model_p, pv, pgt[:,0].view(-1, 1), losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp5-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
    model_p=modtp(layers,losfun)
    model_p = train(epocs, model_p, pv, pgt[:,1].view(-1, 1), losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp5-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')


def exp6(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(), offset:int|None=None, inputDims:int=5):
    flog = open(logfile, 'a+')
    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()
    modtp,layers,losfun = models[exp_no]
    layers[0]=inputDims
    seed=torch.seed()

    for i in range(4):
        pv, pgt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=i)
        yv, ygt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=i, vf='Original Yaw', gtf='Ground Truth Yaw')

        model_p=modtp(layers,losfun)
        print(exp_msg.format(exp_name='exp3'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
        model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp6_iset{i}-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
        model_y=modtp(layers,losfun)
        model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-exp6_iset{i}-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'  +'-epoch{}.model')


def exp7(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(), offset:int|None=None, inputDims:int=5):
    flog = open(logfile, 'a+')
    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()
    modtp,layers,losfun = models[exp_no]
    layers[0]=inputDims

    for i in range(5):
        iset = i if i != 5 else None
        pv, pgt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=iset)
        yv, ygt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=iset, vf='Original Yaw', gtf='Ground Truth Yaw')
        seed=0

        torch.manual_seed(0)
        model_p=modtp(layers,losfun)
        print(exp_msg.format(exp_name='exp3'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
        model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp7_iset{iset}-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{0}'+'-epoch{}.model')
        torch.manual_seed(0)
        model_y=modtp(layers,losfun)
        model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-exp7_iset{iset}-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{0}'  +'-epoch{}.model')


def exp8(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(), offset:int|None=None, inputDims:int=5, seed:int=0):
    flog = open(logfile, 'a+')
    losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()
    modtp,layers,losfun = models[exp_no]
    layers[0]=inputDims

    for i in range(4):
        iset = i 
        pv, pgt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=iset)
        # yv, ygt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=iset, vf='Original Yaw', gtf='Ground Truth Yaw')

        torch.manual_seed(0)
        model_p=modtp(layers,losfun)
        print(exp_msg.format(exp_name='exp3'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
        model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp8_iset{iset}-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
        # torch.manual_seed(0)
        # model_y=modtp(layers,losfun)
        # model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-exp8_iset{iset}-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'  +'-epoch{}.model')
    
    pv, pgt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=None)
    # yv, ygt, _, _ = readCSV_gt_evaled_loo_drivface(infile, inputDims, offset, iset=None, vf='Original Yaw', gtf='Ground Truth Yaw')

    torch.manual_seed(seed)
    model_p=modtp(layers,losfun)
    print(exp_msg.format(exp_name='exp3'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
    model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp8-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
    # torch.manual_seed(0)
    # model_y=modtp(layers,losfun)
    # model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-exp8-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'  +'-epoch{}.model')

trainLog:Any

def exp9(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(),offset:int|None=None,inputDims:int=5, seed:int|None=0):
    try:
        flog = open(logfile, 'a+')
        pv, pgt = readCSV_pitch_and_yaw_together_many_files(datasets[2:],inputDims,offset=offset)

        losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()

        modtp,layers,losfun = models[exp_no]
        assert modtp == CustomNN_YArch
        if inputDims !=5:
            layers[0][0]=2*inputDims
            if layers[0][-1] % 5==0: layers[0][-1]=layers[0][-1]//5*7
            layers[1][0]=layers[0][-1]
        if seed is None: seed = torch.seed()
        torch.manual_seed(seed)
        model_p=modtp(layers,losfun)
        print(exp_msg.format(exp_name='exp5'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
        model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-exp9-pitch_and_yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
    except Exception as e:
        print(f'eroare la modeId:{exp_no} ofs:{offset} idms:{inputDims} with {e}',file=trainLog)

def exp10(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(),offset:int|None=None,inputDims:int=5, seed:int|None=0):
    try:
        flog = open(logfile, 'a+')
        pv, pgt = readCSV_pitch_and_yaw_together_many_files(datasets[2:],inputDims,offset=offset)

        losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()

        modtp,layers,losfun = models[exp_no]
        assert modtp == CustomNN_MArch
        if inputDims !=5:
            layers[0][0]=2*inputDims
            if layers[0][-1] % 5==0: layers[0][-1]=layers[0][-1]//5*7
            layers[1][0]=layers[0][-1]+inputDims
        if seed is None: seed = torch.seed()
        torch.manual_seed(seed)
        model_p=modtp(layers,losfun)
        print(exp_msg.format(exp_name='exp5'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
        model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-expa-pitch_and_yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
    except Exception as e:
        print(f'eroare la modeId:{exp_no} ofs:{offset} idms:{inputDims} with {e}',file=trainLog)


def exp11(exp_no:int, epocs:int, losFn:torch.nn.Module= torch.nn.MSELoss(),offset:int|None=None, inputDims:int=5, seed:int|None=0):
    try:
        flog = open(logfile, 'a+')
        pv, pgt, yv, ygt = readCSV_pitch_and_yaw_many_files(datasets,inputDims,offset=offset)

        pft = (pgt == pv[:,offset]).type(dtype=torch.float32)

        with open(logFolder+f'validexp11.log', 'a+') as validLog:
            print(f'Exp 11 exp_no:{exp_no} losFn:{losFn} inputDims:{inputDims} offset:{offset}\n',pft,file=validLog)

        losName:str = type(losFn).__name__ if not isinstance(losFn,INamedModule) else losFn.name()

        modtp,layers,losfun = models[exp_no]
        layers[0]=inputDims

        if seed is None: seed=torch.seed()

        torch.manual_seed(seed)
        model_p=modtp(layers,losfun)
        model_p = train(epocs, model_p, pv, pgt, losFn, flog, check_points, modelFile+model_p.fun_name+f'-expb-pitch-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')
        torch.manual_seed(seed)
        model_y=modtp(layers,losfun)
        model_y = train(epocs, model_y, yv, ygt, losFn, flog, check_points, modelFile+model_y.fun_name+f'-expb-yaw-input{inputDims}-offset{offset}-losfn{losName}-seed{seed}'+'-epoch{}.model')

        print(exp_msg.format(exp_name='exp1'+model_p.fun_name,tm=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),sed=seed), file=flog)
    except Exception as e:
        print(f'eroare la modeId:{exp_no} ofs:{offset} idms:{inputDims} with {e}',file=trainLog)


def validate2():
    Validation().validateManyModels()


#check_points= [2000, 5000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000]
check_points= [5000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000, 125_000, 150_000]
models= {
         0:(SimpleNN,[5, 100, 1],None),
         1:(SimpleNN,[5, 15, 1],None),
         2:(SimpleNN,[5, 25, 1],None),
         3:(SimpleNN,[5, 25, 15, 1],None),
         4:(SimpleNN,[5, 15, 25, 1],None),
         5:(SimpleNN,[5, 15, 5, 1],None),
         6:(SimpleNN,[5, 15, 1, 1],None),
         7:(SimpleNN,[5, 15, 25, 5, 1],None),
         8:(SimpleNN,[5, 25, 1],torch.nn.Tanh),
         9:(SimpleNN,[5, 25, 15, 1],torch.nn.Tanh),
        10:(SimpleNN,[5, 15, 25, 1],torch.nn.Tanh),
        20:(SimpleNN,[10,25,2],None),
        21:(SimpleNN,[10,25,25,2],None),
        22:(SimpleNN,[10,25,25,2],torch.nn.Tanh),
        23:(SimpleNN,[10,25,25,2],[torch.nn.Tanh,torch.nn.ReLU]),
        24:(SimpleNN,[10,25,25,2],[torch.nn.ReLU,torch.nn.Tanh]),
        25:(SimpleNN,[10,25,45,45,2],None),
        26:(SimpleNN,[10,25,45,45,2],torch.nn.Tanh),
        27:(SimpleNN,[10,25,25,45,45,2],None),
        28:(SimpleNN,[10,25,25,45,2],torch.nn.Tanh),
        29:(SimpleNN,[10,25,25,45,2],[torch.nn.Tanh,torch.nn.Tanh,torch.nn.ReLU]),
        30:(SimpleNN,[10,25,25,45,2],[torch.nn.Tanh,torch.nn.ReLU,torch.nn.ReLU]),

        50:(CustomNN_YArch,[[10,25,10],[10,25,1]],torch.nn.ReLU),
        51:(CustomNN_YArch,[[10,25,25],[25,25,1]],torch.nn.ReLU),

        70:(CustomNN_MArch,[[10,25,10],[15,25,1]],torch.nn.ReLU),
        71:(CustomNN_MArch,[[10,25,25],[30,25,1]],torch.nn.ReLU),
        72:(CustomNN_MArch,[[10,100,25],[30,100,1]],torch.nn.ReLU),
        73:(CustomNN_MArch,[[10,15,1],[6,25,1]],torch.nn.ReLU),

        100:(DeciderNN,[5, 100, 1],None),
        101:(DeciderNN,[5, 15, 1],None),
        102:(DeciderNN,[5, 15, 1],torch.nn.Tanh),
        103:(DeciderNN,[5, 25, 1],None),
        104:(DeciderNN,[5, 25, 1],torch.nn.Tanh),
}



def main_train():
    global err_ok

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    trainLog=open(logFolder+'trainLog.log','a+')
    print(f'--------------------------------------------\n Experiment at {'fst'}',file = trainLog)


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

    #experiment new rerun
    #exp1 - failed to save los fun name
    # exp1(exp_no=1, epocs=30_000)
    # exp1(exp_no=1, epocs=30_000,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=2, epocs=30_000)
    # exp1(exp_no=2, epocs=30_000,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=3, epocs=50_000)
    # exp1(exp_no=3, epocs=50_000,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=7, epocs=30_000)
    # exp1(exp_no=7, epocs=30_000,losFn=CustomLoss_v(err_ok))
    
    # #exp2 - failed to save los fun name
    # exp1(exp_no=1, epocs=30_000, offset=3)
    # exp1(exp_no=1, epocs=30_000, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=7, epocs=30_000, offset=3)
    # exp1(exp_no=7, epocs=30_000, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1, epocs=30_000, offset=2)
    # exp1(exp_no=1, epocs=30_000, offset=2,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=2, epocs=30_000, offset=2)
    # exp1(exp_no=2, epocs=30_000, offset=2,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=4, epocs=50_000, offset=2)
    # exp1(exp_no=4, epocs=50_000, offset=2,losFn=CustomLoss_v(err_ok))

    # print('ex1')
    # exp1(exp_no=1, epocs=30_000, offset=2, inputDims=6)
    # exp1(exp_no=1, epocs=30_000, offset=2, inputDims=6,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=2, epocs=30_000, offset=2, inputDims=6)
    # exp1(exp_no=2, epocs=30_000, offset=2, inputDims=6,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=4, epocs=50_000, offset=2, inputDims=6)
    # exp1(exp_no=4, epocs=50_000, offset=2, inputDims=6,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1, epocs=30_000, offset=3, inputDims=6)
    # exp1(exp_no=1, epocs=30_000, offset=3, inputDims=6,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=2, epocs=30_000, offset=3, inputDims=6)
    # exp1(exp_no=2, epocs=30_000, offset=3, inputDims=6,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=4, epocs=50_000, offset=3, inputDims=6)
    # exp1(exp_no=4, epocs=50_000, offset=3, inputDims=6,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1, epocs=30_000, offset=3, inputDims=7)
    # exp1(exp_no=1, epocs=30_000, offset=3, inputDims=7,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=2, epocs=30_000, offset=3, inputDims=7)
    # exp1(exp_no=2, epocs=30_000, offset=3, inputDims=7,losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=4, epocs=50_000, offset=3, inputDims=7)
    # exp1(exp_no=4, epocs=50_000, offset=3, inputDims=7,losFn=CustomLoss_v(err_ok))

    # print('ex20')
    # exp4(exp_no=20, epocs=125_000)
    # exp4(exp_no=20, epocs=125_000,losFn=CustomLoss_v(err_ok))
    # exp4(exp_no=20, epocs=125_000, offset=3)
    # exp4(exp_no=20, epocs=125_000, offset=3 ,losFn=CustomLoss_v(err_ok))
    # exp4(exp_no=20, epocs=125_000, offset=4, inputDims=6)
    # exp4(exp_no=20, epocs=125_000, offset=4, inputDims=6,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=20, epocs=125_000, offset=3, inputDims=6)
    # # exp4(exp_no=20, epocs=125_000, offset=3, inputDims=6,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=20, epocs=125_000, offset=3, inputDims=7)
    # # exp4(exp_no=20, epocs=125_000, offset=3, inputDims=7,losFn=CustomLoss_v(err_ok))
    # exp4(exp_no=20, epocs=125_000, offset=4)
    # exp4(exp_no=20, epocs=125_000, offset=4,losFn=CustomLoss_v(err_ok))
    # print('ex21')
    # exp4(exp_no=21, epocs=150_000)
    # exp4(exp_no=21, epocs=150_000,losFn=CustomLoss_v(err_ok))
    # exp4(exp_no=21, epocs=150_000, offset=3)
    # exp4(exp_no=21, epocs=150_000, offset=3, losFn=CustomLoss_v(err_ok))
    # exp4(exp_no=21, epocs=150_000, offset=4, inputDims=6)
    # exp4(exp_no=21, epocs=150_000, offset=4, inputDims=6,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=21, epocs=150_000, offset=3, inputDims=6)
    # # exp4(exp_no=21, epocs=150_000, offset=3, inputDims=6,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=21, epocs=150_000, offset=3, inputDims=7)
    # # exp4(exp_no=21, epocs=150_000, offset=3, inputDims=7,losFn=CustomLoss_v(err_ok))

    # # print('ex22')
    # # exp4(exp_no=22, epocs=150_000)
    # # exp4(exp_no=22, epocs=150_000,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=22, epocs=150_000, offset=3)
    # # exp4(exp_no=22, epocs=150_000, offset=3, losFn=CustomLoss_v(err_ok))

    # exp4(exp_no=22, epocs=150_000, offset=4, inputDims=6)
    # exp4(exp_no=22, epocs=150_000, offset=4, inputDims=6,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=22, epocs=150_000, offset=3, inputDims=6)
    # # exp4(exp_no=22, epocs=150_000, offset=3, inputDims=6,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=22, epocs=150_000, offset=3, inputDims=7)
    # # exp4(exp_no=22, epocs=150_000, offset=3, inputDims=7,losFn=CustomLoss_v(err_ok))

    # # print('ex23-24')
    # # exp4(exp_no=23, epocs=150_000)
    # # exp4(exp_no=23, epocs=150_000,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=23, epocs=150_000, offset=3)
    # # exp4(exp_no=23, epocs=150_000, offset=3, losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=24, epocs=150_000)
    # # exp4(exp_no=24, epocs=150_000,losFn=CustomLoss_v(err_ok))
    # # exp4(exp_no=24, epocs=150_000, offset=3)
    # # exp4(exp_no=24, epocs=150_000, offset=3, losFn=CustomLoss_v(err_ok))  

    # # exp4(exp_no=24, epocs=150_000)
    # # exp4(exp_no=24, epocs=150_000,inputDims=7,offset=4)
    # exp4(exp_no=24, epocs=150_000,inputDims=7,offset=5) #9
    # # exp4(exp_no=25, epocs=150_000)
    # # exp4(exp_no=25, epocs=150_000,inputDims=7,offset=4)
    # exp4(exp_no=25, epocs=150_000,inputDims=7,offset=5)
    # # exp4(exp_no=26, epocs=150_000)
    # # exp4(exp_no=26, epocs=150_000,inputDims=7,offset=4)
    # exp4(exp_no=26, epocs=150_000,inputDims=7,offset=5)
    # # exp4(exp_no=27, epocs=150_000)
    # # exp4(exp_no=27, epocs=150_000,inputDims=7,offset=4)
    # exp4(exp_no=27, epocs=150_000,inputDims=7,offset=5)

    # exp4(exp_no=21, epocs=150_000, offset=5, inputDims=7)
    # exp4(exp_no=22, epocs=150_000, offset=5, inputDims=7)

    # exp5(exp_no=22, epocs=100_000, offset=3, inputDims=5) # de refacut experimentele de la 5 care se suprapun cu astea ca de ce nu
    # exp5(exp_no=28, epocs=100_000, offset=3, inputDims=5)
    # exp5(exp_no=29, epocs=100_000, offset=3, inputDims=5)
    # exp5(exp_no=30, epocs=100_000, offset=3, inputDims=5)
    # exp5(exp_no=30, epocs=100_000, offset=3, inputDims=5)

    # exp5(exp_no=22, epocs=100_000, offset=4, inputDims=5) # de refacut experimentele de la 5 care se suprapun cu astea ca de ce nu
    # exp5(exp_no=28, epocs=100_000, offset=4, inputDims=5)
    # exp5(exp_no=29, epocs=100_000, offset=4, inputDims=5)
    # exp5(exp_no=30, epocs=100_000, offset=4, inputDims=5)
    # exp5(exp_no=30, epocs=100_000, offset=4, inputDims=5)

    # exp5(exp_no=22, epocs=100_000, offset=5, inputDims=7)
    # exp5(exp_no=28, epocs=100_000, offset=5, inputDims=7)
    # exp5(exp_no=29, epocs=100_000, offset=5, inputDims=7)
    # exp5(exp_no=30, epocs=100_000, offset=5, inputDims=7)

    # exp5(exp_no=22, epocs=100_000, inputDims=5)
    # exp5(exp_no=28, epocs=100_000, inputDims=5)
    # exp5(exp_no=29, epocs=100_000, inputDims=5)
    # exp5(exp_no=30, epocs=100_000, inputDims=5)

    # # exp5(exp_no=22, epocs=100_000, inputDims=7)
    # # exp5(exp_no=28, epocs=100_000, inputDims=7)
    # # exp5(exp_no=29, epocs=100_000, inputDims=7)
    # # exp5(exp_no=30, epocs=100_000, inputDims=7)

    # exp5(exp_no=22, epocs=100_000, inputDims=7, offset=4)
    # exp5(exp_no=28, epocs=100_000, inputDims=7, offset=4)
    # exp5(exp_no=29, epocs=100_000, inputDims=7, offset=4)
    # exp5(exp_no=30, epocs=100_000, inputDims=7, offset=4)

    # exp5(exp_no=22, epocs=100_000, inputDims=6, offset=3)
    # exp5(exp_no=28, epocs=100_000, inputDims=6, offset=3)
    # exp5(exp_no=29, epocs=100_000, inputDims=6, offset=3)
    # exp5(exp_no=30, epocs=100_000, inputDims=6, offset=3)

    # exp5(exp_no=22, epocs=100_000, inputDims=6, offset=4)
    # exp5(exp_no=28, epocs=100_000, inputDims=6, offset=4)
    # exp5(exp_no=29, epocs=100_000, inputDims=6, offset=4)
    # exp5(exp_no=30, epocs=100_000, inputDims=6, offset=4)

    # 

    # exp1(exp_no=1,epocs=50_000, inputDims=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=5, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=3, losFn=CustomLoss_v(err_ok))

    # exp1(exp_no=1,epocs=50_000, inputDims=5)
    # exp1(exp_no=1,epocs=50_000, inputDims=5, offset=3)
    # exp1(exp_no=1,epocs=50_000, inputDims=6)
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=4)
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=3)
    # exp1(exp_no=1,epocs=50_000, inputDims=7)
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=5)
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=4)
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=3)

    # err_ok=0.01

    # exp1(exp_no=1,epocs=50_000, inputDims=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=5, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=3, losFn=CustomLoss_v(err_ok))

    # err_ok=0.025

    # exp1(exp_no=1,epocs=50_000, inputDims=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=5, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=3, losFn=CustomLoss_v(err_ok))

    # err_ok=0.075

    # exp1(exp_no=1,epocs=50_000, inputDims=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=5, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=6, offset=3, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=5, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=4, losFn=CustomLoss_v(err_ok))
    # exp1(exp_no=1,epocs=50_000, inputDims=7, offset=3, losFn=CustomLoss_v(err_ok))

    # exp1(exp_no=0, epocs=100_000)
    # exp1(exp_no=0, epocs=100_000, offset=2)
    # exp1(exp_no=0, epocs=100_000, inputDims=7)
    # exp1(exp_no=0, epocs=100_000, inputDims=7, offset=5)
    # exp1(exp_no=0, epocs=100_000, inputDims=7, offset=4)

    # exp6(exp_no=0, epocs=100_000)
    # exp6(exp_no=1, epocs=100_000)
    # exp6(exp_no=7, epocs=50_000)
    # exp6(exp_no=8, epocs=50_000)
    # exp6(exp_no=10, epocs=50_000)
    # # exp6(exp_no=0, epocs=50_000, offset=2)
    
    
    # exp6(exp_no=0, epocs=50_000, inputDims=7)
    # exp6(exp_no=0, epocs=50_000, inputDims=7, offset=4)
    # exp6(exp_no=0, epocs=50_000, inputDims=7, offset=5)

    # exp6(exp_no=7, epocs=50_000, inputDims=7)
    # exp6(exp_no=8, epocs=50_000, inputDims=7)
    # exp6(exp_no=10, epocs=50_000, inputDims=7)

    # exp6(exp_no=7, epocs=50_000, inputDims=7, offset=4)
    # exp6(exp_no=8, epocs=50_000, inputDims=7, offset=4)
    # exp6(exp_no=10, epocs=50_000, inputDims=7, offset=4)


    global check_points
    check_points+=[5,10,50,100,200,500]

    # exp7(exp_no=0, epocs=100_000)
    # exp7(exp_no=0, epocs=100_000, offset=2)
    # exp7(exp_no=0, epocs=100_000, inputDims=7)
    # exp7(exp_no=0, epocs=100_000 ,inputDims=7, offset=4)
    # exp7(exp_no=1, epocs=100_000)

    # SimpleNN-layers_7_Tanh_15_Tanh_25_1-exp6_iset3-pitch-input7-offset4-losfnMSELoss-seed5582204839100-epoch5000 - ex 10
    # SimpleNN-layers_5_ReLU_15_ReLU_25_ReLU_5_1-exp6_iset2-pitch-input5-offsetNone-losfnMSELoss-seed397601119700-epoch10000 - ex 7
    # exp8(exp_no=10, epocs=20_000)
    # exp8(exp_no=10, epocs=20_000, offset=2)
    # exp8(exp_no=10, epocs=20_000, inputDims=7)
    # exp8(exp_no=10, epocs=20_000, inputDims=7, offset=4)
    # exp8(exp_no=7, epocs=20_000)
    # exp8(exp_no=7, epocs=20_000, offset=2)
    # exp8(exp_no=7, epocs=20_000, inputDims=7)
    # exp8(exp_no=7, epocs=20_000, inputDims=7, offset=4)

    # exp8(exp_no=10, epocs=20_000, seed=5582204839100)
    # exp8(exp_no=10, epocs=20_000, offset=2, seed=5582204839100)
    # exp8(exp_no=10, epocs=20_000, inputDims=7, seed=5582204839100)
    # exp8(exp_no=10, epocs=20_000, inputDims=7, offset=4, seed=5582204839100)
    # exp8(exp_no=7, epocs=20_000, seed=5582204839100)
    # exp8(exp_no=7, epocs=20_000, offset=2, seed=5582204839100)
    # exp8(exp_no=7, epocs=20_000, inputDims=7, seed=5582204839100)
    # exp8(exp_no=7, epocs=20_000, inputDims=7, offset=4, seed=5582204839100)

    # exp8(exp_no=10, epocs=20_000, seed=397601119700)
    # exp8(exp_no=10, epocs=20_000, offset=2, seed=397601119700)
    # exp8(exp_no=10, epocs=20_000, inputDims=7, seed=397601119700)
    # exp8(exp_no=10, epocs=20_000, inputDims=7, offset=4, seed=397601119700)
    # exp8(exp_no=7, epocs=20_000, seed=397601119700)
    # exp8(exp_no=7, epocs=20_000, offset=2, seed=397601119700)
    # exp8(exp_no=7, epocs=20_000, inputDims=7, seed=397601119700)
    # exp8(exp_no=7, epocs=20_000, inputDims=7, offset=4, seed=397601119700)


    exp11(exp_no=100, epocs=50_000)
    exp11(exp_no=100, epocs=50_000, offset=2)
    exp11(exp_no=100, epocs=50_000, inputDims=7)
    exp11(exp_no=100, epocs=50_000, inputDims=7, offset=4)

    exp11(exp_no=101, epocs=50_000)
    exp11(exp_no=101, epocs=50_000, offset=2)
    exp11(exp_no=101, epocs=50_000, inputDims=7)
    exp11(exp_no=101, epocs=50_000, inputDims=7, offset=4)

    exp11(exp_no=102, epocs=50_000)
    exp11(exp_no=102, epocs=50_000, offset=2)
    exp11(exp_no=102, epocs=50_000, inputDims=7)
    exp11(exp_no=102, epocs=50_000, inputDims=7, offset=4)

    exp11(exp_no=103, epocs=50_000)
    exp11(exp_no=103, epocs=50_000, offset=2)
    exp11(exp_no=103, epocs=50_000, inputDims=7)
    exp11(exp_no=103, epocs=50_000, inputDims=7, offset=4)

    exp11(exp_no=104, epocs=50_000)
    exp11(exp_no=104, epocs=50_000, offset=2)
    exp11(exp_no=104, epocs=50_000, inputDims=7)
    exp11(exp_no=104, epocs=50_000, inputDims=7, offset=4)

    # exp9(exp_no=50,epocs=50_000)
    # exp9(exp_no=50,epocs=50_000, offset=2)
    # exp9(exp_no=50,epocs=50_000, inputDims=7)
    # exp9(exp_no=50,epocs=50_000, inputDims=7, offset=4)
    # exp9(exp_no=51,epocs=50_000)
    # exp9(exp_no=51,epocs=50_000, offset=2)
    # exp9(exp_no=51,epocs=50_000, inputDims=7)
    # exp9(exp_no=51,epocs=50_000, inputDims=7, offset=4)

    # exp10(exp_no=70,epocs=50_000)
    # exp10(exp_no=70,epocs=50_000, offset=2)
    # exp10(exp_no=70,epocs=50_000, inputDims=7)
    # exp10(exp_no=70,epocs=50_000, inputDims=7, offset=4)
    # exp10(exp_no=71,epocs=50_000)
    # exp10(exp_no=71,epocs=50_000, offset=2)
    # exp10(exp_no=71,epocs=50_000, inputDims=7)
    # exp10(exp_no=71,epocs=50_000, inputDims=7, offset=4)
    # exp10(exp_no=72,epocs=50_000)
    # exp10(exp_no=72,epocs=50_000, offset=2)
    # exp10(exp_no=72,epocs=50_000, inputDims=7)
    # exp10(exp_no=72,epocs=50_000, inputDims=7, offset=4)
    # exp10(exp_no=73,epocs=50_000)
    # exp10(exp_no=73,epocs=50_000, offset=2)
    # exp10(exp_no=73,epocs=50_000, inputDims=7)
    # exp10(exp_no=73,epocs=50_000, inputDims=7, offset=4)

    

if __name__ == '__main__':
    main_train()

    validate2()