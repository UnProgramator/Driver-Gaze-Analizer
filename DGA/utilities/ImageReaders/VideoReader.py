from typing import Tuple, override
import cv2
from .IReader import IReader



class VideoReader(IReader):
    video:cv2.VideoCapture
    frame:cv2.typing.MatLike
    ret:bool
    fid=-1
    stop=False
    targetFPS:int|None=None
    crtFps:int|None=None
    skip:int=1

    def __init__(self, videoPath:str, targetFPS:int|None=None):
        self.video = cv2.VideoCapture(videoPath)
        self.targetFPS = targetFPS
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.targetFPS is not None:
            (major_ver, _, _) = (cv2.__version__).split('.')

            if int(major_ver)  < 3 :
                self.crtFps = int(self.video.get(cv2.cv.CV_CAP_PROP_FPS)) # type: ignore # some warning for the variable type was not declared in the cv library
            else :
                self.crtFps = int(self.video.get(cv2.CAP_PROP_FPS))

            self.skip = self.crtFps//self.targetFPS if self.crtFps>self.targetFPS else 1

    def __del__(self):
        self.video.release()
     
    @override
    def getNextFrame(self) -> Tuple[int, cv2.typing.MatLike]:
        if self.stop:
            if self.fid == 0:     # if it is put to stop and the index is 0, then I can go forward. 
                self.stop = False
            else:                 # If the index is not 0 and the stop is true, then I it means I am on the last frame and if I try to go forwards it will overflow
                raise IndexError
        
        if self.skip>1: self.video.set(cv2.CAP_PROP_POS_FRAMES, self.fid+self.skip)

        self.ret, self.frame = self.video.read() # get next frame

        if self.ret:  
            self.fid += self.skip    
            self.frame = self._apply(self.frame)
            return self.fid,self.frame
        else:
            self.stop = True
            raise IndexError
    
    @override
    def getPrevFrame(self)-> Tuple[int, cv2.typing.MatLike]:
        if self.stop:
            if self.fid > 0:     # if it is put to stop and the index is the actual last index (I am yet to know or care), then I can go backward. 
                self.stop = False
            else:                # If the index is 0 and I want to go forwards than it will underflow
                raise IndexError

        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.fid-self.skip) # move to the last captured frame
        self.ret, self.frame = self.video.read() # get that frame

        if self.ret:
            self.fid -= self.skip
            return self.fid,self.frame
        else:
            self.stop = True
            raise IndexError

    @override
    def getCrtFrame(self)-> Tuple[int, cv2.typing.MatLike]:
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
    def reset(self) -> None:
        if self.fid == -1:
            return
        self.fid = -1
        if not self.video.set(cv2.CAP_PROP_POS_FRAMES, 0):
            self.fid = -1
            raise Exception('Unexpted exception')

    def save_frames(self, outPath:str, starting_idx:int=0) -> int:
        '''returnes the next index after the last one used if any given, or the number of frames saved if none is given'''
        idx:int = starting_idx
        for _,im in self:
            succes:bool = cv2.imwrite(outPath.format(idx=idx), im)
            idx+=1
            if not succes:
                raise Exception("couldn't write an immage")
        return idx

