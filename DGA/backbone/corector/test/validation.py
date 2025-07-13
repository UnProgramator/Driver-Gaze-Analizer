import ntpath
import os
import time
from typing import IO
from datetime import datetime

from torch import Tensor
from corectors import validate, loadModel

from CorectionUtilities import *
from SimpleNN import SimpleNN

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

def defaultValidation():
    with open(logFolder+'validationsLogs.log','a+') as lf, open(logFolder+'validationResults.log','a+') as resf:
        default_validateAllModels(folder=modelFile,log=lf,results=resf)


def default_validateAllModels(folder:str|None=None, files:list[str]|None=None, log:IO[str]|None=None, results:IO[str]|None=None, plot_on:bool=False, resCSV:IO[str]|None=None, inputDims:int=5, offset:int|None=None):
    pv4, pgt4, yv4, ygt4 = readCSV_pitch_and_yaw_many_files(datasets,inputDim=inputDims,offset=offset)
    #pv3, pgt3, yv3, ygt3 = readCSV_pitch_and_yaw_many_files(datasets[1:],5)
    pvd, pgtd, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None)
    yvd, ygtd, _, _ = readCSV_gt_evaled_loo_drivface(infile, 5, None, 'Original Yaw', 'Ground Truth Yaw')

    exp_code:str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csvFile:IO[str]

    def __validate_model(filePath:str, valiName:str, vals:torch.Tensor, gt:torch.Tensor):
        print(f'Testing model {filePath} with dataset {valiName}',file=results)
        model=loadModel(filePath,log)
        filename=ntpath.basename(filePath)
        iacc,ialoss,tacc,taloss = validate(model,vals,gt,logFile=results,plotSavefile=(plotfile.format(filename) if plot_on else None) )
        print(exp_code,filename,valiName,iacc,ialoss,tacc,taloss,sep=',',file=csvFile)
        print('--------------------------------------------------------------------\n', file=results)

    def __validate_wrap(filePath:str):
        if 'yaw' in filePath and 'pitch' in filePath:
            __validate_model(filePath,'OURS',yv4,ygt4)
            __validate_model(filePath,'Drivface',yvd,ygtd)
        elif 'yaw' in filePath:
            __validate_model(filePath,'OURS',yv4,ygt4)
            # __validate_model(filePath,'OURS_3',yv3,ygt3)
            __validate_model(filePath,'Drivface',yvd,ygtd)
        elif 'pitch' in filePath:
            __validate_model(filePath,'OURS',pv4,pgt4)
            # __validate_model(filePath,'OURS_3',pv3,pgt3)
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
    csvFile.close()


#ex SimpleNN-layers_14_Tanh_25_Tanh_25_2-exp1-pitch_and_yaw-input7-offset4-losfnMSELoss-epoch150000

class Validation():
    loadedDatasets:dict[tuple[int,int,int],list[tuple[str,tuple[Tensor,Tensor]]]]={}
    expcode:str

    def __init__(self):
        pass

    def __getInfo(self,modelName:str)->dict[str,int|str|None]:
        info:dict[str,int|str|None]={}
        tokens = modelName.split('-')
        dts=tokens[3]
        info['dtype']=3 if dts=='pitch_and_yaw' else 2 if dts=='yaw' else 1
        info['in']=int(tokens[4][5:])
        info['ofs']=int(tokens[5][6:]) if tokens[5][6]!='N' else info['in']-1
        return info


    def __getDataset(self, info:dict[str,int|str|None]) -> list[tuple[str,tuple[Tensor,Tensor]]]:
        tup:tuple[int,int,int] = (info['dtype'],info['in'],info['ofs'])
        if tup not in self.loadedDatasets:
            if tup[0] == 3:
                v,gt = readCSV_pitch_and_yaw_together_many_files(datasets,inputDim=tup[1],offset=tup[2])
                self.loadedDatasets[tup]=[('training',(v,gt))]

                v,gt = readCSV_pitch_and_yaw_together(infile,inputDim=tup[1],offset=tup[2])
                self.loadedDatasets[tup].append(('validation',(v,gt)))
            else:
                tp=(1,tup[1],tup[2])
                ty=(2,tup[1],tup[2])

                pv,pgt,yv,ygt = readCSV_pitch_and_yaw_many_files(datasets,inputDim=tup[1],offset=tup[2])
                self.loadedDatasets[tp]=[('training',(pv,pgt))]
                self.loadedDatasets[ty]=[('training',(yv,ygt))]

                pv,pgt,yv,ygt = readCSV_pitch_and_yaw(infile,inputDim=tup[1],offset=tup[2])
                self.loadedDatasets[tp]=[('validation',(pv,pgt))]
                self.loadedDatasets[ty]=[('validation',(yv,ygt))]


        return self.loadedDatasets[tup]
        

    def validateModel(self,modelPath:str,log:IO[str]|None=None, results:IO[str]|None=None, csvFile:IO[str]|None=None, plotFileTemplate:str|None=None)->None:
            
        model=loadModel(modelPath,log)

        filename=ntpath.basename(modelPath)
        modelName=filename[:filename.find('.')]

        #validationName=?
        info = self.__getInfo(modelName)
        dts = self.__getDataset(info)

        print(info)

        for validationName,t in dts:
            v,gt=t
            pos:list[int]=info['ofs'] if info['dtype'] !=3 else [info['ofs'],info['ofs']+info['in']/2]
            initialv = v[:,pos]
            if info['dtype'] !=3: initialv=initialv.view(-1, 1)
            assert initialv.shape == gt.shape
            iacc,ialoss,tacc,taloss = validate(model=model,inputVals=v,gtVals=gt,initialVals=initialv,datasetName=validationName, logFile=results,
                                               plotSavefile=plotFileTemplate.format(validationName) if plotFileTemplate else None)
            if iacc.shape[0]==1:
                print(self.expcode,modelName,validationName,iacc.item(),ialoss.item(),tacc.item(),taloss.item(),sep=',',file=csvFile)
                print('--------------------------------------------------------------------\n', file=results)
            else:
                print(self.expcode,modelName,validationName+'-pitch',iacc[0].item(),ialoss[0].item(),tacc[0].item(),taloss[0].item(),sep=',',file=csvFile)
                print('--------------------------------------------------------------------\n', file=results)
                print(self.expcode,modelName,validationName+'-yaw',iacc[1].item(),ialoss[1].item(),tacc[1].item(),taloss[1].item(),sep=',',file=csvFile)
                print('--------------------------------------------------------------------\n', file=results)

        print('\n======================================================================\n\n', file=results)

    def validateManyModels(self,folder:str|None=None, files:list[str]|None=None)->None:
        self.expcode = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if files is None:
            if folder is not None:
                files = [folder+f for f in os.listdir(folder)]
            else:
                raise Exception()
        with open(logFolder+'validationsLogs.log','a+') as lf, open(logFolder+'validationResults.log','a+') as resf, open(logFolder+'mean_error_and_loss.csv','w+') as csvFile:
            for f in files:
                self.validateModel(f,lf,resf,csvFile)

    





