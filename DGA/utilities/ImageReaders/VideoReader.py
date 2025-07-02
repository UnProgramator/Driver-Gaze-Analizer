from typing import Final, Tuple, override
import cv2
from .IReader import IReader



class VideoReader(IReader):
    video:cv2.VideoCapture
    frame:cv2.typing.MatLike
    imValid:bool=False
    crtFid=-1
    nextFid=0
    targetFPS:Final[int|None]
    crtFps:int|None=None
    skip:Final[float|None]
    nextfskip:float
    length:int

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

            if self.crtFps > self.targetFPS:
                self.skip = self.crtFps/self.targetFPS
                self.nextfskip=0.0
            else:
                self.skip=None
        else:
            self.skip=None

    def __del__(self):
        self.video.release()
     
    def __incFid(self):
        self.crtFid = self.nextFid
        if self.skip != None:
            self.nextfskip += self.skip
            self.nextFid = int(self.nextfskip)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.nextFid)
        else:
            self.nextFid+=1
        

    def __decFid(self):
        self.crtFid = self.nextFid
        if self.skip != None:
            self.nextfskip -= self.skip
            self.nextFid = int(self.nextfskip)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.nextFid)
        else:
            self.nextFid-=1

    @override
    def getNextFrame(self) -> Tuple[int, cv2.typing.MatLike]:
        if self.nextFid >= self.length:
            print(f'stop{self.nextFid}')
            raise IndexError()
        print(f'{self.nextFid} with id {self.video.get(cv2.CAP_PROP_POS_FRAMES)}')
        self.imValid, self.frame = self.video.read() # get next frame

        if self.imValid:  
            self.__incFid()
            self.frame = self._apply(self.frame)
            return self.crtFid,self.frame
        else:
            if self.nextFid < self.length:
                raise Exception('error has occured when reading frames')
            raise IndexError
    
    @override
    def getPrevFrame(self)-> Tuple[int, cv2.typing.MatLike]:
        if self.nextFid <=0:
            raise IndexError()

        self.imValid, self.frame = self.video.read() # get that frame

        if self.imValid:
            self.__decFid()
            return self.crtFid,self.frame
        else:
            if self.nextFid >=0:
                raise Exception('error has occured when reading frames')
            raise IndexError()

    @override
    def getCrtFrame(self)-> Tuple[int, cv2.typing.MatLike]:
        if self.imValid:
            return self.crtFid,self.frame
        else:
            raise IndexError()

    @override
    def getCrtIndex(self)-> int:
        if self.imValid:
            return self.crtFid
        else:
            raise IndexError()
    
    @override
    def reset(self) -> None:
        if self.crtFid == -1:
            return
        self.imValid = False
        self.crtFid = -1
        if self.skip is not None:
            self.nextfskip = 0.0
        if not self.video.set(cv2.CAP_PROP_POS_FRAMES, 0):
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

