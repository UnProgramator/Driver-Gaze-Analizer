from typing import Iterable, Final
from typing_extensions import override

from .InputPathGeneratorReader import InputPathGeneratorReader

class FolderParserPathGenerator(InputPathGeneratorReader, Iterable[str]):
    dir:Final[str]

    def __init__(self, directory:str):
        self.dir = directory

    @override
    def get_next_image_path(self) -> str: 
        raise NotImplementedError

    @override
    def get_prev_image_path(self) -> str: 
        raise NotImplementedError

    @override
    def get_crt_image_path(self) -> str: 
        raise NotImplementedError

    @override
    def reset(self) -> None:
        raise NotImplementedError