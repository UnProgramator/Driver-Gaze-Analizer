from operator import contains
from typing import Any, Final
import cv2
from l2cs.results import GazeResultContainer
import numpy as np
import random
import tkinter.filedialog

import pandas

from backbone import my_pipeline


# Parameters

batch = 2 # 1-4

basePath = 'D:/DCIM/images_12fps/'

inTestFile = basePath+'pitch_and_yaw_b{}.csv'.format(batch)
gtTestFile = basePath+'gt_b{}.csv'.format(batch)
imgDir     = basePath+'poze_test_batch_{}/'.format(batch)
gtWndName:Final[str] = 'Ground Truth'
imgWndName:Final[str] = 'Current Farme'


Debug:bool = True
frameHeight = 600         # viewport height
frameWidth = 1000        # viewport width
lrMargin = 10         # left-right margin
distBetweenPoints  = 20         # distance between points
scale = 100     # scaling pitch on window hight
delta = 0.003   # intercative correction dispalcement
VMAX = 2.0      # max values for pitch and yaw
N = frameWidth//distBetweenPoints        # max. number of point in viewport
totalFrames = 0    # number of frames (points on curve)
crtFrame = 0    #curent frame to be processed

pv   = 0        # current point in vieport, in 0 .. N-1

color1:Final = (0, 102, 204)   # graph color
thickness1:Final = 1  # graph line tickness

radius2:Final = 3     # point radius
color2:Final = (0, 179, 255)   # point color
thickness2:Final = 2  # point line tickness

color3:Final = (128, 128, 0)   # gt graph color
thickness3:Final = 1  # gt graph line tickness

radius4:Final = 5     # gt point radius
color4:Final = (255, 20, 20)   # gt point color
thickness4:Final = 2  # gt point line tickness

radius5:Final = 9     # crt point radius
color5:Final = (0, 0, 255)   # crt point color
thickness5:Final = 2  # crt point line tickness

pitch:list[str]
yaw:list[str]

gtp:list[str]
gty:list[str]

names:list[str]

bd:np.ndarray[Any,np.dtype[np.float32]]|None = None

def main():
    global totalFrames, pcrt, pv,pitch,yaw,gtp,gty
    random.seed()

    ########## MAIN()
    print('Gaze labeling tool v0.2 started ...')

    # reading csw file
    inFileName:str
    gtFileName:str
    (inFileName,gtFileName) = getFiles()

    #reads a fucking csv file, cahnge to use py-csv
    print('Extracting pitch and yaw values from '+inFileName+'.')
    
    print('read l2cs output from file')
    pitch,yaw = readInFile(inFileName,True)

    print('read gt from file')
    try:
        gtp,gty=readGt(gtFileName,True)
    except:
        gtp = pitch.copy()     # Ground Truth point list
        gty = yaw.copy()

    if len(gtp) == 0:
        gtp = pitch.copy()     # Ground Truth point list
        gty = yaw.copy()

    #diplay window graph
    cv2.namedWindow(gtWndName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(gtWndName, frameWidth, frameHeight)
    bg = np.zeros((frameHeight, frameWidth, 3), np.uint8)
    bg[::] = 255
    cv2.imshow(gtWndName, bg)
    display(gtWndName, imgWndName, bg, pitch, gtp)

    # main loop
    isPitch:bool = True
    global pcrt, pv, crtFrame
    while True:
        k = cv2.waitKeyEx(0)

        if k == 27: # esc
            saveGt(gtFileName)
            cv2.destroyAllWindows()
            print('Finish')
            exit(0)

        elif k==2424832 or k == ord('a'):      #no more arrow - a   # arrows - platform dependent (works on windows)
            #if Debug: print('left')

            if crtFrame>0:
                crtFrame -= 1
            else:
                print('Leftmost position!')

            if pv>0:
                pv -= 1  
 

        elif k==2555904 or k == ord('d'):
            #if Debug: print('right')

            if crtFrame<totalFrames-1:
                crtFrame += 1
            else:
                print('Rightmost position!')

            if pv < N-1:
                pv += 1

        elif k== 2490368 or k == ord('w'):
            #if (DEBUG): print('up')
            if isPitch:
                if float(gtp[crtFrame])<VMAX:
                    gtp[crtFrame] = str(float(gtp[crtFrame]) + delta + random.random()/500)
            else:
                if float(gty[crtFrame])<VMAX:
                    gty[crtFrame] = str(float(gty[crtFrame]) + delta + random.random()/500)
        elif k== 2621440 or k == ord('s'):
            #if (DEBUG): print('down')
            if isPitch:
                if float(gtp[crtFrame])>-VMAX:
                    gtp[crtFrame] = str(float(gtp[crtFrame]) - delta - random.random()/500)
            else:
                if float(gty[crtFrame])>-VMAX:
                    gty[crtFrame] = str(float(gty[crtFrame]) - delta - random.random()/500)
        
        elif k == ord('q'):
            if crtFrame > 0:
                gtp[crtFrame]=gtp[crtFrame-1]
                gty[crtFrame]=gty[crtFrame-1]
                print(f'clone frame pitch and yaw {crtFrame} with {crtFrame-1}')
            else:
                print('Error, can\'t copy the state of previouse frame for frame 0');

        elif k == ord('e'):
            if crtFrame < totalFrames-1:
                gtp[crtFrame]=gtp[crtFrame+1]
                gty[crtFrame]=gty[crtFrame+1]
                print(f'clone frame pitch and yaw {crtFrame} with {crtFrame+1}')
            else:
                print('Error, can\'t copy the state of previouse frame for frame 0');

        elif k == ord('y'):
            saveGt(gtFileName)

        elif k == ord('c'):             # c
            print('Switching to {}!'.format('Pitch' if isPitch else 'Yaw'))
            isPitch = not isPitch  

        elif k == ord('r'):      #r - reset either pitch or yaw if needed
            #if Debug: print('left')
            if isPitch:
                gtp[crtFrame] = pitch[crtFrame]
                print('reset pitch')
            else:
                gty[crtFrame] = yaw[crtFrame]
                print('reset yaw')

        elif k == ord('o'):
            isPitch = True
            if crtFrame != 0:
                crtFrame = 0
                pv = 0
                pcrt = 0
                print('reset index to 0')
            else:
                crtFrame = totalFrames-1
                pv = N-1
                pcrt = totalFrames-N
                print('reset index to 0')

        elif k == ord('A'):
            if crtFrame>=100:
                crtFrame -= 100
            else:
                crtFrame = 0
            pv = 0

        elif k == ord('D'):
            if crtFrame<totalFrames-100:
                crtFrame += 100
            else:
                crtFrame=totalFrames-1
            pv = N-1


        if isPitch:
            display(gtWndName, imgWndName, bg, pitch, gtp, '  -  Pitch')
        else:
            display(gtWndName, imgWndName, bg, yaw, gty, '  -  Yaw')

def saveGt(gtFilePath:str):
    global gtp, gty
    with open(gtFilePath,'w+') as fid:
        print('Ground Truth Pitch,Ground Truth Yaw',file=fid)
        for i in range(len(gtp)):
            point1 = gtp[i]
            point2 = gty[i]
            print(point1,',',point2,file=fid,sep=None)
    print(f'Ground Truth saved in {gtFilePath} !')

def readInFile(inFileName:str,normalCsv:bool=False) -> tuple[list[str],list[str]]:
    pitch:list[str]
    yaw:list[str]
    global totalFrames, names, bd
    if not normalCsv:
        with open(inFileName,'r') as inFile:
            line = inFile.readline().rstrip()
            pitch = line.split(',')
            pitch = pitch[1:]
            
            print(totalFrames, 'frames - pitch: ', pitch)
            line = inFile.readline().rstrip()
            yaw = line.split(',')
            yaw = yaw[1:]
            assert len(yaw)==totalFrames, 'Invalid number of YAW values. (<> with PITCH values)!'    
            print(totalFrames, 'frames - yaw: ', yaw)
    else:
        pd = pandas.read_csv(inFileName,dtype='str')
        pitch = pd['Pitch']
        yaw = pd['Yaw']
        names = [i.strip() for i in pd['FileName']]
        if contains(pd,'B1'):

            B1:list[float] = [float(i) for i in pd['B1']]
            B2:list[float] = [float(i) for i in pd['B2']]
            B3:list[float] = [float(i) for i in pd['B3']]
            B4:list[float] = [float(i) for i in pd['B4']]
            bd = np.array([[B1[i],B2[i],B3[i],B4[i]] for i in range(len(B1))], dtype=np.float32)
    
    totalFrames = len(pitch)
    assert len(yaw)==totalFrames, 'Invalid number of YAW values. (<> with PITCH values)!'   
    print('number of frames',totalFrames)
    return pitch,yaw

def readGt(gtFileName:str,normalCsv:bool=False) -> tuple[list[str],list[str]]:
    gtp:list[str]
    gty:list[str]
    if not normalCsv:
        #open gtFile (if exists)
        gtFile = open(gtFileName,'r')
        line = gtFile.readline().rstrip()
        gtp = line.split(',')
        gtp = gtp[1:]
        line = gtFile.readline().rstrip()
        gtFile.close()
        gty = line.split(',')
        gty = gty[1:] 
    
        return gtp,gty
    else:
        pd = pandas.read_csv(gtFileName,dtype='str')
        gtp = pd['Ground Truth Pitch']
        gty = pd['Ground Truth Yaw']
        return gtp,gty


def getFiles() -> tuple[str,str]:
    '''
        get the files from the predefined path or using tkinter windows
    '''
    inFileName:str
    gtFileName:str
    if Debug:
        inFileName:str = inTestFile
        gtFileName:str = gtTestFile
    else:
        root = tkinter.Tk()
        inFileName = tkinter.filedialog.askopenfilename(initialdir='.', title='Select input CSV file.', filetypes=(('CSV files', '*.csv;*.txt'), ('all files', '*.*')))
        root.destroy()
        root = tkinter.Tk()
        gtFileName = tkinter.filedialog.askopenfilename(initialdir='.', title='Select GroundTruth CSV file.', filetypes=(('CSV files', '*.csv;*.txt'), ('all files', '*.*')))
        root.destroy()
    return inFileName, gtFileName

def display(wnd:str, frameWnd:str, window:cv2.typing.MatLike, m:list[str], g:list[str], im_info:str=''):
    global crtFrame, pv, N, totalFrames, gty, gtp, pitch, yaw, H
    startingPointInFrame = crtFrame - pv
    nrp = min(N, totalFrames - startingPointInFrame)
    window[::] = 255
    start_point = (0, frameHeight//2)
    end_point   = (frameWidth, frameHeight//2)
    cv2.line(window, start_point, end_point, (102,102,0), 1)
    cv2.putText(window, names[crtFrame]+im_info
                ,(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    for i in range(nrp-1):
        cv2.putText(window, str(startingPointInFrame+i), (i*distBetweenPoints + lrMargin, frameHeight-10), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.3, (102,102,0), 1, cv2.LINE_AA)

    # print the points and the lines
    end_point1=(0,0)
    end_point2=(0,0)
    for i in range(nrp-1):
        p1:float= float(m[startingPointInFrame+i])
        p2:float = float(m[startingPointInFrame+i+1])
        start_point1 = (i*distBetweenPoints + lrMargin, int(frameHeight//2 - p1*scale))
        end_point1   = ((i+1)*distBetweenPoints + lrMargin, int(frameHeight//2 - p2*scale))
        window = cv2.line(window, start_point1, end_point1, color1, thickness1)
        window = cv2.circle(window,start_point1, radius2, color2, thickness2)
        p1 = float(g[startingPointInFrame+i])
        p2 = float(g[startingPointInFrame+i+1])
        start_point2 = (i*distBetweenPoints + lrMargin, int(frameHeight/2 - p1*scale))
        end_point2   = ((i+1)*distBetweenPoints + lrMargin, int(frameHeight/2 - p2*scale))
        window = cv2.line(window, start_point2, end_point2, color3, thickness3)
        window = cv2.circle(window,start_point2, radius4, color4, thickness4)

    window = cv2.circle(window,end_point1, radius2, color2, thickness2)
    window = cv2.circle(window,end_point2, radius4, color4, thickness4)
    p3 = float(g[crtFrame]) if crtFrame < totalFrames else 0.0
    crt_point = (pv*distBetweenPoints + lrMargin, int(frameHeight/2 - p3*scale))
    window = cv2.circle(window, crt_point, radius5, color5, thickness5)
    cv2.imshow(wnd, window)
    #framePath = imgDir + 'image_'+str(crtFrame)+'.png'
    framePath = imgDir+names[crtFrame]

    image = cv2.imread(framePath) 
    cv2.putText(image, framePath, (0,15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))

    pt1:cv2.typing.Point=(image.shape[0]//2,image.shape[1]//2)

    leng=50
    if bd is None:
        dy:float = leng * (np.sin(float(gtp[crtFrame])) if np.abs(float(gtp[crtFrame]))<np.pi else np.sin(float(gtp[crtFrame]))*np.abs(float(gtp[crtFrame])))
        dx:float = leng * (np.sin(float(gty[crtFrame])) if np.abs(float(gty[crtFrame]))<np.pi else np.sin(float(gty[crtFrame]))*np.abs(float(gty[crtFrame])))
        pt2:cv2.typing.Point=np.round((pt1[0]-dy,pt1[1]-dx)).astype(int)
        cv2.arrowedLine(image,pt1, 
                              pt2,
                              color=(0,255,0),thickness=5)

        dy = leng * np.sin(float(pitch[crtFrame]))
        dx= leng * np.sin(float(yaw[crtFrame]))
        pt2:cv2.typing.Point=np.round((pt1[0]-dy,pt1[1]-dx)).astype(int)
        cv2.arrowedLine(image,pt1, 
                              pt2,
                              color=(0,0,255),thickness=5)
    else:
        x= GazeResultContainer(None, None, None, None, None)

        x.bboxes=np.array([bd[crtFrame]],dtype=np.float32)

        x.pitch=np.array([float(gtp[crtFrame])],dtype=np.float32)
        x.yaw=np.array([float(gty[crtFrame])],dtype=np.float32)

        my_pipeline.my_pipeline.my_render(image,x,(0,522,0))

        x.pitch=np.array([float(pitch[crtFrame])],dtype=np.float32)
        x.yaw=np.array([float(yaw[crtFrame])],dtype=np.float32)

        my_pipeline.my_pipeline.my_render(image,x)



    cv2.imshow(frameWnd, image)


if __name__ == '__main__':
    Debug=True
    main()