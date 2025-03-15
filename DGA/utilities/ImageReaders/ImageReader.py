from typing import override
from .IReader import IReader
from PathGenerators.InputPathGeneratorReader import InputPathGeneratorReader


class ImageReager(IReader):
    imReader:InputPathGeneratorReader|None = None
    fid=-1

    def __init__(self, imReader:InputPathGeneratorReader|None = None):
        self.imReader = imReader

    @override
    def getCrtIndex(self)-> int:
        if self.fid != -1:
            return self.fid
        else:
            raise IndexError