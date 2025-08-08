import os
import pandas as pd


base_file = 'D:/DCIM/images_12fps/pitch_and_yaw_bdrivface.csv'
gt_file = 'D:/DCIM/images_12fps/gt_bdrivface.csv'
out_file = 'D:/DCIM/results/data/Err_gaze.csv'
impa='D:/Programming/Datasets/drivface/DrivFace/DrivImages/DrivImages/'
Header = 'Frame,Original Pitch,Ground Truth Pitch,Pitch Err,Original Yaw,Ground Truth Yaw,Yaw Err'
headerBase='Frame id,Yaw,Pitch,FileName'
headerGT='Ground Truth Pitch,Ground Truth Yaw,Frame Err'

def catFile(base:str, gt:str,out:str)->None:
    f = pd.read_csv(base)
    g = pd.read_csv(gt)

    p:list[float] = f['Pitch']
    y:list[float] = f['Yaw']
    pg:list[float] = g['Ground Truth Pitch']
    yg:list[float] = g['Ground Truth Yaw']
    be:list[float] = g['Frame Err']

    

    with open(out,'w+') as fileDesc:
        print(Header, file=fileDesc)
        for i in range(len(p)):
            print(i+1,p[i],pg[i],abs(p[i]-pg[i]),y[i],yg[i],abs(y[i]-yg[i]), be[i],
                    file = fileDesc,
                    sep=',')

def decatFile():
    src = pd.read_csv(out_file)
    p:list[float] = src['Original Pitch']
    y:list[float] = src['Original Yaw']
    pg:list[float] = src['Ground Truth Pitch']
    yg:list[float] = src['Ground Truth Yaw']
    # be:list[float] = g['Frame Err']

    file = [f for f in os.listdir(impa) if f.find('jpg')!=-1 and not f == '20130529_01_Driv_096_f .jpg']

    assert len(file)==len(p)

    with open(base_file,'w+') as base, open(gt_file,'w+') as gt:
        print(headerBase, file=base)
        print(headerGT, file=gt)
        for i in range(len(p)):
            print(i, y[i], p[i],file[i],sep=',',file=base)
            print(pg[i], yg[i],'0',sep=',',file=gt)


def main():
    print('start')
    decatFile()


if __name__=='__main__':
    main()
