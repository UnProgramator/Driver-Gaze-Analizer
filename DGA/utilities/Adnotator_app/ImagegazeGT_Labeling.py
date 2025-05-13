import cv2
import numpy as np
import random
import tkinter.filedialog


# Parameters

inTestFile = "D:/Home/dan/doc/cs/doc/pub/Deiana_ISETC2024/Code/Labeling/pitch_and_yaw.csv"
gtTestFile = "D:/Home/dan/doc/cs/doc/pub/Deiana_ISETC2024/Code/Labeling/gt.csv"
imgDir = "D:/Home/dan/doc/cs/doc/pub/Deiana_ISETC2024/Results/imgs/"
gtWnd:str = "Ground Truth"
imgWnd:str = "Current Farme"


DEBUG:bool = True
H = 600         # viewport height
W = 1000        # viewport width
D1 = 10         # left-right margin
D  = 20         # distance between points
scale = 100     # scaling pitch on window hight
delta = 0.003   # intercative correction dispalcement
VMAX = 2.0      # max values for pitch and yaw
N = int(W/D)    # max. number of point in viewport
nrFrames = 0    # number of frames (points on curve)
pcrt = 0        # current point in 0 .. nrFrames-1
pv   = 0        # view point in 0 .. N-1
color1 = (0, 102, 204)   # graph color
thickness1 = 1  # graph line tickness
radius2 = 3     # point radius
color2 = (0, 179, 255)   # point color
thickness2 = 2  # point line tickness
color3 = (128, 128, 0)   # gt graph color
thickness3 = 1  # gt graph line tickness
radius4 = 5     # gt point radius
color4 = (255, 20, 20)   # gt point color
thickness4 = 2  # gt point line tickness
radius5 = 9     # crt point radius
color5 = (0, 0, 255)   # crt point color
thickness5 = 2  # crt point line tickness
random.seed()

def display(wnd:str, frameWnd:str, img:cv2.typing.MatLike, m, g):
    nrp = min(N, nrFrames - pcrt)
    img[::] = 255
    crtFrame = pcrt+pv
    start_point = (0, int(H/2))
    end_point   = (W, int(H/2))
    img = cv2.line(img, start_point, end_point, (102,102,0), 1)
    for i in range(nrp-1):
        img = cv2.putText(img, str(pcrt+i), (i*D + D1, H-10), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.3, (102,102,0), 1, cv2.LINE_AA)
    for i in range(nrp-1):
        p1 = float(m[pcrt+i])
        p2 = float(m[pcrt+i+1])
        start_point1 = (i*D + D1, int(H/2 - p1*scale))
        end_point1   = ((i+1)*D + D1, int(H/2 - p2*scale))
        img = cv2.line(img, start_point1, end_point1, color1, thickness1)
        img = cv2.circle(img,start_point1, radius2, color2, thickness2)
        p1 = float(g[pcrt+i])
        p2 = float(g[pcrt+i+1])
        start_point2 = (i*D + D1, int(H/2 - p1*scale))
        end_point2   = ((i+1)*D + D1, int(H/2 - p2*scale))
        img = cv2.line(img, start_point2, end_point2, color3, thickness3)
        img = cv2.circle(img,start_point2, radius4, color4, thickness4)
    img = cv2.circle(img,end_point1, radius2, color2, thickness2)
    img = cv2.circle(img,end_point2, radius4, color4, thickness4)
    p3 = float(g[crtFrame])
    crt_point = (pv*D + D1, int(H/2 - p3*scale))
    img = cv2.circle(img, crt_point, radius5, color5, thickness5)
    cv2.imshow(wnd, img)
    framePath = imgDir + "image_"+str(crtFrame)+".png"
    image = cv2.imread(framePath) 
    cv2.imshow(frameWnd, image)

def start():
    ########## MAIN()
    print("Gaze labeling tool v1.0 started ...")

    # reading csw file
    if DEBUG:
        inFileName:str = inTestFile
        gtFileName:str = gtTestFile
    else:
        root = tkinter.Tk()
        inFileName = tkinter.filedialog.askopenfilename(initialdir=".", title="Select input CSV file.", filetypes=(("CSV files", "*.csv;*.txt"), ("all files", "*.*")))
        root.destroy()
        root = tkinter.Tk()
        gtFileName = tkinter.filedialog.askopenfilename(initialdir=".", title="Select GroundTruth CSV file.", filetypes=(("CSV files", "*.csv;*.txt"), ("all files", "*.*")))
        root.destroy()

    print("Extracting pitch and yaw values from "+inFileName+".")
    inFile = open(inFileName,'r')
    line = inFile.readline().rstrip()
    pitch = line.split(",")
    pitch = pitch[1:]
    nrFrames = len(pitch)
    print(nrFrames, "frames - pitch: ") #, pitch)
    line = inFile.readline().rstrip()
    yaw = line.split(",")
    yaw = yaw[1:]
    assert len(yaw)==nrFrames, "Invalid number of YAW values. (<> with PITCH values)!"    
    print(nrFrames, "frames - yaw: ") #, yaw)
    inFile.close()

    #open gtFile (if exists)
    try:
        gtFile = open(gtFileName,'r')
        line = gtFile.readline().rstrip()
        gtp = line.split(",")
        gtp = gtp[1:]
        line = gtFile.readline().rstrip()
        gtFile.close()
        gty = line.split(",")
        gty = gty[1:]    
    except:
        gtp = pitch.copy()     # Ground Truth point list
        gty = yaw.copy()

    #diplay window graph
    cv2.namedWindow(gtWnd, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(gtWnd, W, H)
    bg = np.zeros((H, W, 3), np.uint8)
    bg[::] = 255
    cv2.imshow(gtWnd, bg)
    display(gtWnd, imgWnd, bg, pitch, gtp)

    # main loop
    isPitch = 1
    while(1):
        k = cv2.waitKeyEx(50)
        cg = 0
        if k == 27:             # ESC
            if(not isPitch):
                break;
            print("Switching to YAW!")
            isPitch = 0
            pv = 0
            pcrt = 0
            display(gtWnd, imgWnd, bg, yaw, gty)        
        if(k==2424832):         # arrows - platform dependent (works on windows)
            #if (DEBUG): print("left")
            if(pv>0):
                pv = pv - 1   
                cg = 1
            else:
                if(pcrt > 0):
                    pcrt = pcrt - 1
                    cg = 1
                else:
                    print("Leftmost position!")
        if(k==2555904):
            #if (DEBUG): print("right")
            if pv < N-1:
                pv = pv + 1            
                cg = 1
            else:
                if pcrt < nrFrames-N-1:
                    pcrt = pcrt +1
                    cg = 1              
                else:
                    print("Rightmost position!")
        if(k== 2490368):
            #if (DEBUG): print("up")
            if(isPitch):
                if float(gtp[pcrt+pv])<VMAX:
                    gtp[pcrt+pv] = str(float(gtp[pcrt+pv]) + delta + random.random()/500)
            else:
                if float(gty[pcrt+pv])<VMAX:
                    gty[pcrt+pv] = str(float(gty[pcrt+pv]) + delta + random.random()/500)
            cg = 1
        if(k== 2621440):
            #if (DEBUG): print("down")
            if(isPitch):
                if float(gtp[pcrt+pv])>-VMAX:
                    gtp[pcrt+pv] = str(float(gtp[pcrt+pv]) - delta - random.random()/500)
            else:
                if float(gty[pcrt+pv])>-VMAX:
                    gty[pcrt+pv] = str(float(gty[pcrt+pv]) - delta - random.random()/500)
            cg = 1
        if cg:
            if(isPitch):
                display(gtWnd, imgWnd, bg, pitch, gtp)
            else:
                display(gtWnd, imgWnd, bg, yaw, gty)
            #if (DEBUG): print(pcrt, pv)

    # exit - flush gtFile and free all resources
    cv2.destroyAllWindows()
    gtFile = open(gtFileName,'w')
    line = "Ground Truth Pitch"
    for point in gtp:
        line = line+ ", " + str(point)
    gtFile.write(line+"\n")
    line = "Ground Truth Yaw"
    for point in gty:
        line = line+ ", " + str(point)
    gtFile.write(line)
    gtFile.close()
    print("Ground Truth saved in "+gtFileName+"!")
    print("Bye!")