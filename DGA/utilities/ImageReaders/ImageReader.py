from typing import Tuple
from typing import Tuple, override
import numpy as np
import cv2
from .IReader import IReader
from PathGenerators.InputPathGeneratorReader import InputPathGeneratorReader


class ImageReager(IReader):
    imReader:InputPathGeneratorReader = None
    fid=-1

    def __init__(self, imReader:InputPathGeneratorReader = None):
        self.imReader = imReader
    
    @override
    def getNextFrame(self) -> Tuple[int, cv2.MatLike]:
        pass

    @override
    def getPRevFrame(self)-> Tuple[int, cv2.MatLike]:
        pass

    @override
    def getCrtFrame(self)-> Tuple[int, cv2.MatLike]:
        pass

    @override
    def getCrtIndex(self)-> int:
        if self.fid != -1:
            return self.fid
        else:
            raise IndexError