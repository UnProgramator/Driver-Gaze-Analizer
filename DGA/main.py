from multiprocessing import process

from cv2.typing import MatLike
from sklearn.cluster import KMeans

from utilities.ImageReaders.ImageReader import ImageReager
from utilities.ImageReaders.VideoReader import VideoReader
from utilities.PathGenerators.DrivfaceInput import DrivfaceInput
from backbone.clustering import clustering
from backbone.my_pipeline import my_pipeline
import os
import torch
import numpy as np
import cv2

from backbone.processor import Processor
from utilities.PathGenerators.TemplatePathGenerator import TemplatePathGenerator

#from utilities.Validation.dreyeve_validation import drvalidation


def darken(img:MatLike) -> MatLike:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # value = -80 #whatever value you want to add
    value = -50 #whatever value you want to add
    hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
    image:MatLike = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image

def lighten(img:MatLike) -> MatLike:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = 30 #whatever value you want to add
    hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
    image:MatLike = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image

def gproc():
    return Processor(-0.65, 0.2, -0.5, 0.8, 2, 30 )

def f1(imgsrc):
    

    proc = gproc()

    cod = proc.process(imgsrc).split_words().codate_aparitions()
    
    
    print(*proc.get_action_list())
    print()
    for w in proc.get_words(): print(w)
    print()
    print(*cod)

def f4(imgsrc):
    proc = gproc()
    sum = 0
    ar = proc.validate(imgsrc)
    for pair in ar:
        print(pair[0],',',pair[1],',',pair[2],',',pair[3])
        
        if pair[1][0]==pair[2] or pair[1][1]==pair[2]:
            sum += 1
        
    print(sum / len(ar))

def f2(imgsrc):
    proc = gproc()

    cod = proc.process(imgsrc).reduce_noise().split_words().codate_aparitions()
    
    
    print(*proc.get_action_list())
    print()
    for w in proc.get_words(): print(w)
    print()
    for w in cod: print(w)
    print()
    
    words = np.array(proc.codate_aparitions())

    kmean = KMeans(n_clusters=2, random_state=0, n_init="auto")
    m = kmean.fit(words)
    print(m.labels_)
    
    words2 = np.array(proc.codate_duration())
    
    kmean2 = KMeans(n_clusters=2, random_state=0, n_init="auto")
    m2 = kmean2.fit(words2)
    print(m2.labels_)

def f3(imgsrc):
    proc = gproc()

    cod = proc.process(imgsrc)
    print(proc.get_action_list())
    print()

def my_videos_test():

    proc= gproc()

    pat = r'D:\DCIM\DCIMC/'

    sidx = 0


    for i in range(4,6):
        imInput = VideoReader(pat+'MOVC{:04d}.avi'.format(i))
        imInput.setFilters([darken])
        sidx += proc.render(imInput, 'D:/test/im_{imNr:04d}.png')
    
# Frame,
# Original Pitch,Ground Truth Pitch,Pitch Err
# Original Yaw,  Ground Truth Yaw,  Yaw Err

def video_parameters():
    target_framerate:int|None = None
    track_no:int = 4
    video_no:list[int] = []

    csv_header = 'Frame,Original Pitch,Ground Truth Pitch,Pitch Err,Original Yaw,Ground Truth Yaw,Yaw Err'

    proc:Processor = gproc()

    paths = r'D:\DCIM\DCIMC{nr_t}/'
    video_form = r'D:\DCIM\DCIMC{nr_t}/MOVC{nr_v:04d}.avi'
    outFile = r'D:\DCIM\results.csv'

    with open(outFile, 'w') as csv_file:

        csv_frid = 1
        print(csv_header,file=csv_file)

        for t in range(track_no):
            for v in range(video_no[t]):
                imIn = VideoReader(video_form.format(nr_t=t, nrv = v), target_framerate)
            
                vals = proc.get_pitch_yaw_list(imIn)

                for _, y, p in vals:
                    print(csv_frid, p, 0, 0, y, 0, 0,file=csv_file)


def im_save():
    video_form = r'D:\DCIM\DCIMC{nr_t}/MOVC{nr_v:04d}.avi'
    outFolder = 'D:\\DCIM\\images\\'
    img_form = r'_{idx:05d}.png'
    target_framerate=12

    dcim_nr=['','1','2','3']
    dcim_sn=[(0,6), (2,8) ,(0,13),(0,5)]


    letter = ['A', 'B', 'C', 'D']

    for i in range(4):
        nr = dcim_nr[i]
        sn = dcim_sn[i]
        idx=0
        for v in range(sn[0],sn[1]):
            imIn = VideoReader(video_form.format(nr_t=nr, nr_v = v), target_framerate)
            imIn.setFilters([darken])
            idx = imIn.save_frames(outFolder + letter[i] + img_form, idx)
        print(letter[i]+str(idx))
        

def im_save2():
    video_form = r'D:\DCIM\DCIMC{nr_t}/MOVC{nr_v:04d}.avi'
    outFolder = 'D:\\DCIM\\poze_test\\'
    img_form = r'_{idx:05d}.png'
    target_framerate=12

    dcim_nr=['','1','2','3']
    dcim_sn=[(0,6), (2,8) ,(0,13),(0,5)]


    letter = ['A', 'B', 'C', 'D']

    i=0

    for i in range(1):
        nr = dcim_nr[i]
        sn = dcim_sn[i]
        idx=0
        for v in range(0,1):
            imIn = VideoReader(video_form.format(nr_t=nr, nr_v = v), target_framerate)
            imIn.setFilters([darken])
            idx = imIn.save_frames(outFolder + letter[i] + img_form, idx)

        print(letter[i]+str(idx))

def my_im_test():
    dire = r'D:\DCIM\poze_test'
    pt = dire + r'\A_{:05d}.png'

    tpg = TemplatePathGenerator(pt, 84, 1346)

    imps = ImageReager(tpg)

    pc = gproc()

    pc.render(imps)
    

    

def main():
    
    #imgsrc = DrivfaceInput((1,2,3))
    
    # f4(imgsrc)
    # print("3A 4V")
    # imgsrc = DrivfaceInput((1,2,3,4))
    
    # f4(imgsrc)

    #drvalidation()

    #f1(imgsrc) 

    #my_videos_test()

    #im_save2()
    my_im_test()

    return 0

if __name__ == "__main__":
    main()
