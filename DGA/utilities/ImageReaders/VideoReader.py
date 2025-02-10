from typing import Iterable, Self, Tuple, override
import numpy as np
import cv2
from .IReader import IReader


class VideoReader(IReader):
    video:cv2.VideoCapture = None
    frame:cv2.MatLike = None
    ret:bool = None
    fid=-1
    stop=False

    def __init__(self, videoPath:str):
        self.video = cv2.VideoCapture(videoPath)
     
    @override
    def getNextFrame(self) -> Tuple[int, cv2.MatLike]:
        if self.stop:
            if self.fid == 0:     # if it is put to stop and the index is 0, then I can go forward. 
                self.stop = False
            else:                 # If the index is not 0 and the stop is true, then I it means I am on the last frame and if I try to go forwards it will overflow
                raise IndexError

        self.ret, self.frame = self.video.read() # get next frame
        self.fid += 1

        if self.ret:    
            return self.fid,self.frame
        else:
            self.stop = True
            raise IndexError
    
    @override
    def getPrevFrame(self)-> Tuple[int, cv2.MatLike]:
        if self.stop:
            if self.fid > 0:     # if it is put to stop and the index is the actual last index (I am yet to know or care), then I can go backward. 
                self.stop = False
            else:                # If the index is 0 and I want to go forwards than it will underflow
                raise IndexError

        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.fid-1) # move to the last captured frame
        self.ret, self.frame = self.video.read() # get that frame

        if self.ret:
            self.fid -= 1
            return self.fid,self.frame
        else:
            self.stop = True
            raise IndexError

    @override
    def getCrtFrame(self)-> Tuple[int, cv2.MatLike]:
        if self.ret and not self.stop:
            return self.fid,self.frame
        else:
            self.stop = True
            raise IndexError

    @override
    def getCrtIndex(self)-> int:
        if self.fid != -1:
            return self.fid
        else:
            raise IndexError
    
    @override
    def reset(self):
        if self.fid == -1:
            return
        self.fid = 0
        if not self.video.set(cv2.CAP_PROP_POS_FRAMES, 0):
            self.fid = -1
            raise Exception('Unexpted exception')

    def __iter__(self) -> Self:
        self.reset()
        return self

    def __next__(self)-> Tuple[int, cv2.MatLike]:
        return self.getNextFrame()