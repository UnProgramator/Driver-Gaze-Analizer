from math import e
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

experiment='e8'

prefix=f'{experiment}-'

plotfile:str  = resultsPath+"plots/{}.png"
logFolder = resultsPath+f'{prefix}logs/'
logfile:str   = logFolder+'results.log'
modelFolder:str = resultsPath+f'{prefix}models/'

dataPath=resultsPath+'data/'

infile:str = dataPath+'err_bdrivface.csv'

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
        default_validateAllModels(folder=modelFolder,log=lf,results=resf)


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
    loadedDatasets:dict[tuple[int,int,int,int],list[tuple[str,tuple[Tensor,Tensor]|Tensor]]]={}
    expcode:str
    trainDataset:str|list[str]
    validDataset:str|list[str]

    def __init__(self):
        pass

    def __getInfo(self,modelName:str)->dict[str,int|str|None]:
        info:dict[str,int|str|None]={}
        tokens = modelName.split('-')
        struct=tokens[1].split('_')
        dts=tokens[3]
        info['indim']=int(tokens[4][5:])
        info['inlayer']=int(struct[1])
        info['out']=int(struct[-1])
        info['ofs']=int(tokens[5][6:]) if tokens[5][6]!='N' else info['indim']-1
        info['dtype']=3 if dts=='pitch_and_yaw' else 2 if dts=='yaw' else 1
        #backwards Compatible
        if tokens[2].find('_')>=0: 
            if tokens[2].find('iset'):
                info['iset']=int(tokens[2][9]) if tokens[2][9] != 'N' else 9
        elif tokens[-1].find('iset'):
            vset=tokens[-1].split('_')[1]
            info['iset']=int(vset) if vset[0] != 'N' else 9
        elif tokens[-1].find('exclude'):
            vset=tokens[-1].split('_')[1]
            info['exclude']=int(vset)
        # match tokens[0]:
        #     case 'DeciderNN':
        #         info['GT']='eq'
        
        # print(f'file{modelName:>50} -tp{info['dtype']} -indim{info['indim']} -inlayer{info['inlayer']} -of{info['ofs']}')
        return info

    def preproces(self, info, vals, gt, offset):
        if 'GT' in info:
            match info['GT']:
                case 'eq':
                    ref = vals[:,offset].view(-1,1)
                    dif = gt == ref
                    gt = dif.type(dtype=torch.float32)

        return vals, gt


    def __getDataset(self, info:dict[str,int|str|None]) -> list[tuple[str,tuple[Tensor,Tensor]]]:
        tup:tuple[int,int,int,int] = (info['dtype'] if 'iset' not in info else info['dtype']*10+info['iset'],info['indim'],info['ofs'],info['inlayer'])


        if tup not in self.loadedDatasets:
            if tup[0] == 3: # OK
                # dd = readCSV_pitch_and_yaw_together_many_files(datasets[2:],inputDim=tup[1],offset=tup[2])
                # v,gt = dd[FieldNames.Vals], dd[FieldNames.GT]
                # self.loadedDatasets[tup]=[('training',(v,gt))]
                # self.loadedDatasets[tup]= [(f'{FieldNames.fErr}-t', dd[FieldNames.fErr])]

                # dd = readCSV_pitch_and_yaw_together_many_files(datasets[:2],inputDim=tup[1],offset=tup[2])
                # v,gt = dd[FieldNames.Vals], dd[FieldNames.GT]
                # self.loadedDatasets[tup]=[('validation',(v,gt))]
                # self.loadedDatasets[tup]= [(f'{FieldNames.fErr}-v', dd[FieldNames.fErr])]

                dd = readCSV_pitch_and_yaw_together_many_files(datasets,inputDim=tup[1],offset=tup[2])
                v,gt = dd[FieldNames.Vals], dd[FieldNames.GT]
                self.loadedDatasets[tup]=[('training',(v,gt))]
                self.loadedDatasets[tup].append((f'{FieldNames.fErr}-t', dd[FieldNames.fErr]))


                # v,gt = readCSV_pitch_and_yaw_together(infile,inputDim=tup[1],offset=tup[2])
                # pv1,pgt1,_,_ = drivface_readPitch(infile,inputDim=tup[1],offset=tup[2])
                # yv1,ygt1,_,_ = drivface_readYaw  (infile,inputDim=tup[1],offset=tup[2])
                # v = torch.cat((pv1,yv1),dim=1)
                # gt = torch.cat((pgt1,ygt1),dim=1)

                dd1 = readCSV_drivface_sep(infile,inputDim=tup[1],offset=tup[2],bCat=True)
                v = torch.cat((dd1[FieldNames.pVals],dd1[FieldNames.yVals]), dim=1)
                gt = torch.cat((dd1[FieldNames.pGT],dd1[FieldNames.yGT]), dim=1)
                self.loadedDatasets[tup].append(('testing',(v,gt)))
                self.loadedDatasets[tup].append((f'{FieldNames.fErr}-v', dd1[FieldNames.fErr]))
            else:
                if 'iset' not in info: #OK
                    tp=(1,tup[1],tup[2],tup[3])
                    ty=(2,tup[1],tup[2],tup[3])
                    if tup[1]==tup[3]:
                        if 'exclude' not in info:
                            dd = readCSV_pitch_and_yaw_many_files(datasets,inputDim=tup[1],offset=tup[2])
                            pv,pgt,yv,ygt = dd[FieldNames.pVals], dd[FieldNames.pGT], dd[FieldNames.yVals], dd[FieldNames.yGT]
                            pv,pgt = self.preproces(info,pv,pgt,tup[2])
                            yv,ygt = self.preproces(info,yv,ygt,tup[2])
                            self.loadedDatasets[tp]=[('training',(pv,pgt))]
                            self.loadedDatasets[ty]=[('training',(yv,ygt))]
                            self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-t', dd[FieldNames.fErr])]
                            self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-t', dd[FieldNames.fErr])]
                        else:
                            dd = readCSV_pitch_and_yaw_many_files(datasets[1:],inputDim=tup[1],offset=tup[2])
                            vd = readCSV_pitch_and_yaw(datasets[0],inputDim=tup[1],offset=tup[2])
                            pv,pgt,yv,ygt = dd[FieldNames.pVals], dd[FieldNames.pGT], dd[FieldNames.yVals], dd[FieldNames.yGT]
                            pvv,pgtv,yvv,ygtv = vd[FieldNames.pVals], vd[FieldNames.pGT], vd[FieldNames.yVals], vd[FieldNames.yGT]
                            pv,pgt = self.preproces(info,pv,pgt,tup[2])
                            yv,ygt = self.preproces(info,yv,ygt,tup[2])
                            self.loadedDatasets[tp]=[('training-1',(pv,pgt))]
                            self.loadedDatasets[ty]=[('training-1',(yv,ygt))]
                            self.loadedDatasets[tp]=[('validation-1',(pvv,pgtv))]
                            self.loadedDatasets[ty]=[('validation-1',(yvv,ygtv))]
                            self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-t', dd[FieldNames.fErr])]
                            self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-t', dd[FieldNames.fErr])]


                        #pv1,pgt1,yv1,ygt1 = readCSV_pitch_and_yaw(infile,inputDim=tup[1],offset=tup[2])
                        # pv1,pgt1,_,_ = drivface_readPitch(infile,inputDim=tup[1],offset=tup[2])
                        # yv1,ygt1,_,_ = drivface_readYaw  (infile,inputDim=tup[1],offset=tup[2])

                        dd1 = readCSV_drivface_sep(infile,inputDim=tup[1],offset=tup[2],bCat=True)
                        # pv1,pgt1=self.preproces(info,pv1,pgt1,tup[2])
                        # yv1,ygt1=self.preproces(info,yv1,ygt1,tup[2])
                        pv1, pgt1, yv1, ygt1 = dd1[FieldNames.pVals], dd1[FieldNames.pGT], dd1[FieldNames.yVals], dd1[FieldNames.yGT]
                        self.loadedDatasets[tp]+=[('testing',(pv1,pgt1))]
                        self.loadedDatasets[ty]+=[('testing',(yv1,ygt1))]
                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-v', dd1[FieldNames.fErr])]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-v', dd1[FieldNames.fErr])]
                    else: 
                        dd = readCSV_pitch_and_yaw_together_many_files(datasets,inputDim=tup[1],offset=tup[2])
                        v,gt = dd[FieldNames.Vals], dd[FieldNames.GT]
                        pgt=gt[:,0].view(-1,1)
                        ygt=gt[:,1].view(-1,1)
                        pv,pgt=self.preproces(info,v,pgt,tup[2])
                        yv,ygt=self.preproces(info,v,ygt,tup[2])
                        self.loadedDatasets[tp]=[('training',(pv,pgt))]
                        self.loadedDatasets[ty]=[('training',(yv,ygt))]
                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-t', dd[FieldNames.fErr])]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-t', dd[FieldNames.fErr])]

                        # v,gt = readCSV_pitch_and_yaw_together(infile,inputDim=tup[1],offset=tup[2])
                        # pv1,pgt1,_,_ = drivface_readPitch(infile,inputDim=tup[1],offset=tup[2])
                        # yv1,ygt1,_,_ = drivface_readYaw  (infile,inputDim=tup[1],offset=tup[2])

                        dd1 = readCSV_drivface_sep(infile,inputDim=tup[1],offset=tup[2],bCat=True)
                        pv1, pgt1, yv1, ygt1= dd1[FieldNames.pVals], dd1[FieldNames.pGT], dd1[FieldNames.yVals], dd1[FieldNames.yGT]
                        v1 = torch.cat((pv1,yv1),dim=1)
                        pv1,pgt1=self.preproces(info,v1,pgt1,tup[2])
                        yv1,ygt1=self.preproces(info,v1,ygt1,tup[2])
                        print('='*80,'\nError: Driovface values not read properly\n','='*80)
                        self.loadedDatasets[tp]+=[('testing',(v1,pgt1))]
                        self.loadedDatasets[ty]+=[('testing',(v1,ygt1))]
                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-v', dd1[FieldNames.fErr])]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-v', dd1[FieldNames.fErr])]
                else:
                    tp=(10+info['iset'],tup[1],tup[2],tup[3])
                    ty=(20+info['iset'],tup[1],tup[2],tup[3])
                    if info['iset'] != 9: # OK
                        
                        # pvt,pgtt,pvv,pgtv = drivface_readPitch(infile,inputDim=tup[1],offset=tup[2],iset=info['iset'])
                        # yvt,ygtt,yvv,ygtv =   drivface_readYaw(infile,inputDim=tup[1],offset=tup[2],iset=info['iset'])

                        dd1 = readCSV_drivface_sep(infile,inputDim=tup[1],offset=tup[2],bCat=False)
                        pv1, pgt1, yv1, ygt1, fErr= dd1[FieldNames.pVals], dd1[FieldNames.pGT], dd1[FieldNames.yVals], dd1[FieldNames.yGT], dd1[FieldNames.fErr]

                        iset=info['iset']

                        pvv = pv1.pop(iset)
                        pvt = torch.cat(pv1)
                        pgtv = pgt1.pop(iset)
                        pgtt = torch.cat(pgt1)

                        yvv = yv1.pop(iset)
                        yvt = torch.cat(yv1)
                        ygtv = ygt1.pop(iset)
                        ygtt = torch.cat(pgt1)

                        ferv = fErr.pop(iset)
                        fert = torch.cat(fErr)

                        self.loadedDatasets[tp]=[(f'training-{info['iset']}',(pvt,pgtt))]
                        self.loadedDatasets[ty]=[(f'training-{info['iset']}',(yvt,ygtt))]

                        self.loadedDatasets[tp]+=[(f'validation-{info['iset']}',(pvv,pgtv))]
                        self.loadedDatasets[ty]+=[(f'validation-{info['iset']}',(yvv,ygtv))]

                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-st{info['iset']}', fert)]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-st{info['iset']}', fert)]

                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-sv{info['iset']}', ferv)]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-sv{info['iset']}', ferv)]

                        dd = readCSV_pitch_and_yaw_many_files(datasets,inputDim=tup[1],offset=tup[2])
                        pv,pgt,yv,ygt = dd[FieldNames.pVals], dd[FieldNames.pGT], dd[FieldNames.yVals], dd[FieldNames.yGT]
                        self.loadedDatasets[tp]+=[('testing',(pv,pgt))]
                        self.loadedDatasets[ty]+=[('testing',(yv,ygt))]
                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-v', dd[FieldNames.fErr])]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-v', dd[FieldNames.fErr])]
                    else: # OK
                        dd = readCSV_pitch_and_yaw_many_files(datasets,inputDim=tup[1],offset=tup[2])
                        pv,pgt,yv,ygt = dd[FieldNames.pVals], dd[FieldNames.pGT], dd[FieldNames.yVals], dd[FieldNames.yGT]
                        self.loadedDatasets[tp]=[('testing',(pv,pgt))]
                        self.loadedDatasets[ty]=[('testing',(yv,ygt))]
                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-v', dd[FieldNames.fErr])]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-v', dd[FieldNames.fErr])]

                        # v,gt = readCSV_pitch_and_yaw_together(infile,inputDim=tup[1],offset=tup[2])
                        # pv1,pgt1,_,_ = drivface_readPitch(infile,inputDim=tup[1],offset=tup[2])
                        # yv1,ygt1,_,_ = drivface_readYaw  (infile,inputDim=tup[1],offset=tup[2])

                        dd1 = readCSV_drivface_sep(infile,inputDim=tup[1],offset=tup[2],bCat=True)
                        pv1, pgt1, yv1, ygt1= dd1[FieldNames.pVals], dd1[FieldNames.pGT], dd1[FieldNames.yVals], dd1[FieldNames.yGT]
                        
                        print('='*80,'\nError: Driovface values not read properly\n','='*80)
                        self.loadedDatasets[tp]+=[('training',(pv1,pgt1))]
                        self.loadedDatasets[ty]+=[('training',(yv1,ygt1))]
                        self.loadedDatasets[tp]+= [(f'{FieldNames.fErr}-t', dd1[FieldNames.fErr])]
                        self.loadedDatasets[ty]+= [(f'{FieldNames.fErr}-t', dd1[FieldNames.fErr])]

        return self.loadedDatasets[tup]
   

    def __getType(self,info)->str:
        if info['dtype'] in [1,2,3]:
            return 'pitch' if info['dtype']==1 else 'yaw'
        else:
            return 'pitch' if info['dtype']//10==1 else 'yaw'
    
    def __getFErr(self,name:str,datasets:list[tuple[str,torch.Tensor|tuple[torch.Tensor,torch.Tensor]]])->torch.Tensor:     
        match name.split('-'):
            case ['validation']:
                fn=f'{FieldNames.fErr}-v'
            case ['training']:
                fn=f'{FieldNames.fErr}-t'
            case ['validation','u']:
                fn=f'{FieldNames.fErr}-v'
            case ['training',x]:
                fn=f'{FieldNames.fErr}-st{x}'
            case ['validation',x]:
                fn=f'{FieldNames.fErr}-sv{x}'
            case _:
                raise Exception('critical unforseen error: wtf has happened? change part of the code but not all?')
        for p,t in datasets:
            if p==fn:
                return t
        raise Exception('critical unforseen error: wtf has happened? change part of the code but not all?')

    def validateModel(self,modelPath:str,log:IO[str]|None=None, results:IO[str]|None=None, csvFile:IO[str]|None=None, plotFileTemplate:str|None=None)->None:
            
        model=loadModel(modelPath,log)

        filename=ntpath.basename(modelPath)
        modelName=filename[:filename.rfind('.')]

        info = self.__getInfo(modelName)
        dts = self.__getDataset(info)

        print(info)

        testType=self.__getType(info)

        for validationName,t in dts:
            if validationName.find(FieldNames.fErr) >= 0:
                continue

            fErt:torch.Tensor=self.__getFErr(validationName, dts)

            v,gt=t
            pos:int|list[int]
            if info['dtype'] !=3:
                if info['indim']==info['inlayer']:
                    pos=info['ofs']  
                else:
                    if info['dtype'] == 2:
                        pos=info['ofs']+info['indim']
                    else:
                        pos=info['ofs']
            else:
               pos=[info['ofs'],info['ofs']+info['indim']]
            initialv = v[:,pos]
            if type(pos) is int: 
                initialv=initialv.view(-1, 1)
            assert initialv.shape == gt.shape, f'initial values tensor shape ({initialv.shape}) is not right (the as as the shape of the gt ({gt.shape}))'
            iacc,ipac,inac,ialoss,tacc,tpac,tnac,taloss = validate(model=model,inputVals=v,gtVals=gt,initialVals=initialv,datasetName=validationName, logFile=results, mask=fErt,
                                                                    plotSavefile=plotFileTemplate.format(validationName) if plotFileTemplate else None)

            if info['out'] == 1:
                print(experiment,self.expcode,modelName,validationName,testType,
                      iacc.item(),ipac.item(),inac.item(),ialoss.item(),tacc.item(),tpac.item(),tnac.item(),taloss.item(),tacc.item()-iacc.item(),sep=',',file=csvFile)
                print('--------------------------------------------------------------------\n', file=results)
            else:
                print(experiment,self.expcode,modelName,validationName+'-pitch','pitch',iacc[0].item(),ipac[0].item(),inac[0].item(),ialoss[0].item(),tacc[0].item(),tpac[0].item(),tnac[0].item(),taloss[0].item(),tacc[0].item()-iacc[0].item(),sep=',',file=csvFile)
                print(experiment,self.expcode,modelName,validationName+'-yaw',  'yaw',  iacc[1].item(),ipac[1].item(),inac[1].item(),ialoss[1].item(),tacc[1].item(),tpac[1].item(),tnac[1].item(),taloss[1].item(),tacc[1].item()-iacc[1].item(),sep=',',file=csvFile)
                print('--------------------------------------------------------------------\n', file=results)

        print('\n======================================================================\n\n', file=results)

    def validateManyModels(self,folder:str|None=None, files:list[str]|None=None)->None:
        global logFolder,modelFolder
        self.expcode = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if files is None:
            if folder is not None:
                files = [folder+f for f in os.listdir(folder) if f.find('model')>0]
            else:
                print('Use backup folder')
                files = [modelFolder+f for f in os.listdir(modelFolder) if f.find('model')>0]
        with open(logFolder+'validationsLogs.log','a+') as lf, open(logFolder+'validationResults.log','a+') as resf, open(logFolder+'mean_error_and_loss.csv','w+') as csvFile:
            print('expid,exptimestamp,model,validset,validation,iacc,ipac,inac,ialoss,tacc,tpac,tnac,taloss,improvment',file=csvFile)
            for f in files:
                self.validateModel(f,lf,resf,csvFile)

    





