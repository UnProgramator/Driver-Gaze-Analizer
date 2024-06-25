from calendar import c
from multiprocessing import process

from sklearn.cluster import KMeans
from imageReaders.DrivfaceInput import DrivfaceInput
from backbone.clustering import clustering
from backbone.my_pipeline import my_pipeline
import os
import torch
import numpy as np

from backbone.processor import Processor

from utilities.Validation.dreyeve_validation import drvalidation


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
    
    
    

def main():
    
    # imgsrc = DrivfaceInput((1,2,3))
    
    # f4(imgsrc)
    # print("3A 4V")
    imgsrc = DrivfaceInput((1,2,3,4))
    proc = gproc()
    #proc.render(imgsrc)
    proc.render(imgsrc, savePath=r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\trying\image_{imNr}.png')
    
    #f4(imgsrc)

    # drvalidation()

    return 0

if __name__ == "__main__":
    main()
