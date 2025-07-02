import pandas as pd


base_file = 'D:/DCIM/pitch_and_yaw_b{}.csv'
gt_file = 'D:/DCIM/gt_b{}.csv'
out_file = 'D:/DCIM/err_b{}.csv'
Header = 'Frame,Original Pitch,Ground Truth Pitch,Pitch Err,Original Yaw,Ground Truth Yaw,Yaw Err'

def catFile(base:str, gt:str,out:str)->None:
    f = pd.read_csv(base)
    g = pd.read_csv(gt)

    p:list[float] = f['Pitch']
    y:list[float] = f['Yaw']
    pg:list[float] = g['Ground Truth Pitch']
    yg:list[float] = g['Ground Truth Yaw']

    with open(out,'w+') as fileDesc:
        print(Header, file=fileDesc)
        for i in range(len(p)):
            print(i+1,p[i],pg[i],abs(p[i]-pg[i]),y[i],yg[i],abs(y[i]-yg[i]),
                    file = fileDesc,
                    sep=',')


def main():
    for i in range(1,5):
        catFile(base_file.format(i), gt_file.format(i), out_file.format(i))

if __name__=='__main__':
    main()