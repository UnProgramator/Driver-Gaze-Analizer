from imageReaders.DrivfaceInput import DrivfaceInput
from backbone.clustering import clustering
from backbone.my_pipeline import my_pipeline
import os
import torch
import numpy as np

from backbone.processor import Processor

def f1(imgsrc):
    

    proc = Processor(-0.65, 0.01, -0.8, 0.8, 2, 30 )

    cod = proc.process(imgsrc).split_words().codate_aparitions()
    
    
    print(*proc.get_action_list())
    print()
    for w in proc.get_words(): print(w)
    print()
    print(*cod)


def f2(imgsrc):
    proc = Processor(-0.65, 0.01, -0.8, 0.8, 2, 30 )

    cod = proc.process(imgsrc).reduce_noise().split_words().codate_aparitions()
    
    
    print(*proc.get_action_list())
    print()
    for w in proc.get_words(): print(w)
    print()
    print(*cod)


def main():
    
    imgsrc = DrivfaceInput((1,))

    f1(imgsrc)
    
    f2(imgsrc)

    return 0

if __name__ == "__main__":
    main()
