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
        self.crt_idx = self.low


    @override
    def get_next_image_path(self) -> str:
        if self.crt_idx >= self.high:
            raise IndexError()

        ct_id = self.crt_idx
        self.crt_idx += 1

        #raise NotImplemented() # verify index validity

        return self.temp_path.format(ct_id)

    @override
    def get_prev_image_path(self) -> str:
        if self.crt_idx <= self.low:
            raise IndexError()

        ct_id = self.crt_idx
        self.crt_idx -= 1

        #raise NotImplemented() # verify index validity

        return self.temp_path.format(ct_id)

    @override
    def get_crt_image_path(self):
        return self.temp_path.format(self.crt_idx)

    @override
    def reset(self):
        self.crt_idx = self.low

    