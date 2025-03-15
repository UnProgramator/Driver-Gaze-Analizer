from collections.abc import Iterable
from cv2.typing import MatLike

class InputPathGeneratorReader:
    def get_next_image_path(self) -> str: raise NotImplementedError
    def images(self, returnPath:bool = False) -> Iterable[MatLike]: raise NotImplementedError