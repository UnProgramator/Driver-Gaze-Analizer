from backbone.my_pipeline import my_pipeline
from l2cs import render
import cv2
import os
import torch
from backbone.processor import Processor

from imageReaders.DrivfaceInput import DrivfaceInput

    
def main():

    imgsrc = DrivfaceInput((1,2,3,4))

    proc = Processor(-0.65, 0.2, -0.2, 0.4, 2, 30 )

    print("pipeline created succesfully")
    
    
    proc.render(imgsrc, savePath='C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\image_{imNr}.png')


if __name__ == "__main__":
    main()