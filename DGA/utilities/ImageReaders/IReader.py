from types import NotImplementedType
from typing import Tuple
from cv2.typing import MatLike

class IReader:
    def getNextFrame(self) -> Tuple[int, MatLike]: raise NotImplementedType()
    def getPRevFrame(self)-> Tuple[int, MatLike]:  raise NotImplementedType()
    def getCrtFrame(self)-> Tuple[int, MatLike]:   raise NotImplementedType()
    def getCrtIndex(self)-> int: raise NotImplementedType()

    def reset(self): raise NotImplementedType()