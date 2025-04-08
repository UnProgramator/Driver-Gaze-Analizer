import os
from typing import Iterable, Final
from typing_extensions import override
from .InputPathGeneratorReader import InputPathGeneratorReader

class FileListGenerator(InputPathGeneratorReader, Iterable[str]):
    dr:Final[str]
    files:Final[list[str]]
    crtIm:int=-1

    def __init__(self, directory:str, fileList:list[str]):
        self.dr = directory
        self.files=fileList

    @override
    def get_next_image_path(self) -> str: 
        if self.crtIm >= len(self.files)-1:
            raise IndexError()
        self.crtIm+=1
        return self.dr + self.files[self.crtIm]


    @override
    def get_prev_image_path(self) -> str: 
        if self.crtIm <= 0:
            raise IndexError()
        self.crtIm-=1
        return self.dr + self.files[self.crtIm]

    @override
    def get_crt_image_path(self) -> str: 
        if self.crtIm < 0 or self.crtIm>=len(self.files):
            raise IndexError()
        return self.dr + self.files[self.crtIm]

    @override
    def reset(self) -> None:
        self.crtIm=-1
    
    @staticmethod
    def parseDirectory(directory:str):
        files:list[str]= os.listdir(directory)
        return FileListGenerator(directory=directory, fileList=files)

    @staticmethod
    def parseDirectoryFromFile(directory:str, fileName:str='pics.txt', file_sameDir:bool=True):
        files:list[str]
        fl = directory+fileName if file_sameDir else fileName
        with open(fl,'r') as fdesc:
            files= fdesc.readlines()
        return FileListGenerator(directory = directory, fileList=files)
        