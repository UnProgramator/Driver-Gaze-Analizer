from utilities.ImageReaders.IReader import IReader
from utilities.ImageReaders.VideoReader import VideoReader
import cv2


def adnotatorMain(indir:str, outdir:str='./adnotations-default/adnotation.text'): # trebuie refacut pentru a include sa se adnoteze si poze
    reader:IReader = VideoReader(indir)
    while True:
        try:
            id,fr = reader.getNextFrame()
            cv2.imshow('Video', fr)
            k = cv2.waitKey(20)
            if k == 'Q':
                break
        except Exception as e:
            break


if __name__ == '__main__':
    adnotatorMain('')
        