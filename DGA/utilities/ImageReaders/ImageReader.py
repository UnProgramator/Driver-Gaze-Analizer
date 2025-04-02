from typing import Tuple, override

import cv2
from cv2.typing import MatLike
from .IReader import IReader
from PathGenerators.InputPathGeneratorReader import InputPathGeneratorReader


class ImageReager(IReader):
    imReader:InputPathGeneratorReader
    fid:int=-1
    im:MatLike|None = None

    def __init__(self, imReader:InputPathGeneratorReader):
        self.imReader = imReader

    @override
    def getCrtIndex(self)-> int:
        if self.fid != -1:
            return self.fid
        else:
            raise IndexError
    
    @override
    def getNextFrame(self) -> Tuple[int, MatLike]:
        path = self.imReader.get_next_image_path()
        self.im = cv2.imread(path)
        self.fid += 1

        return self.fid, self.im

    @override
    def getPrevFrame(self)-> Tuple[int, MatLike]: 
        path = self.imReader.get_prev_image_path()
        self.im = cv2.imread(path)
        self.fid -= 1

        return self.fid, self.im

    @override
    def getCrtFrame(self)-> Tuple[int, MatLike]:
        if self.im is not None and self.im != -1:
            return self.fid, self.im
        raise Exception("No frame was read OR invalid curent frame/index")

    @override
    def reset(self):
        self.imReader.reset()