

from collections.abc import Iterable


class ImageReader:
    def get_next_image_path(self) -> str:
        pass
    
    def images(self, returnPath:bool = False) -> Iterable:
        pass