from multiprocessing import process

from cv2.typing import MatLike
from sklearn.cluster import KMeans

from utilities.ImageReaders.VideoReader import VideoReader
from utilities.PathGenerators.DrivfaceInput import DrivfaceInput
from backbone.clustering import clustering
from backbone.my_pipeline import my_pipeline
import os
import torch
import numpy as np

from backbone.processor import Processor

#from utilities.Validation.dreyeve_validation import drvalidation


def darken(img:MatLike) -> MatLike:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = -80 #whatever value you want to add
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
    
 

def main():
    
    #imgsrc = DrivfaceInput((1,2,3))
    
    # f4(imgsrc)
    # print("3A 4V")
    # imgsrc = DrivfaceInput((1,2,3,4))
    
    # f4(imgsrc)

    #drvalidation()

    #f1(imgsrc) 

    my_videos_test()

    return 0

if __name__ == "__main__":
    main()
