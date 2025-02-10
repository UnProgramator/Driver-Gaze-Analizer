

from collections.abc import Iterable


class InputPathGeneratorReader:
    def get_next_image_path(self) -> str:
        pass
    
    def images(self, returnPath:bool = False) -> Iterable:
        pass