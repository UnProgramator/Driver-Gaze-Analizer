import os
from typing import Iterable, Final
from typing_extensions import override

from .InputPathGeneratorReader import InputPathGeneratorReader

class TemplatePathGenerator(InputPathGeneratorReader, Iterable[str]):
    
    temp_path:str
    low:Final[int]
    high:Final[int]
    crt_idx:int

    def __init__(self, path_template:str, end_a:int, end_b:int|None=None):
        self.temp_path = path_template
        if end_b is not None:
            self.low = end_a
            self.high = end_b
        else:
            self.low = 0
            self.high = end_a
        self.crt_idx = self.low-1


    @override
    def get_next_image_path(self) -> str:
        fl:str
        lidx:Final[int] = self.crt_idx
        while True:
            if self.crt_idx >= self.high-1:
                self.crt_idx = lidx
                raise IndexError()
            self.crt_idx += 1
            fl = self.temp_path.format(self.crt_idx) # verify index validity   
            if os.path.isfile(fl):
                return fl

    @override
    def get_prev_image_path(self) -> str:
        fl:str
        lidx:Final[int] = self.crt_idx
        while True:
            if self.crt_idx <= self.low+1:
                self.crt_idx = lidx
                raise IndexError()
            self.crt_idx -= 1
            fl = self.temp_path.format(self.crt_idx) # verify index validity
            if os.path.isfile(fl):
                return fl

    @override
    def get_crt_image_path(self):
        if self.crt_idx < self.low or self.crt_idx > self.high:
            raise IndexError()
        return self.temp_path.format(self.crt_idx)

    @override
    def reset(self):
        self.crt_idx = self.low-1

    