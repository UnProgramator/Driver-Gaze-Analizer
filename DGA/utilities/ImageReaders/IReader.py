from typing import Callable, List, Self, Tuple
from types import NotImplementedType
from cv2.typing import MatLike

class IReader:
    def getNextFrame(self) -> Tuple[int, MatLike]: raise NotImplementedType()
    def getPrevFrame(self)-> Tuple[int, MatLike]:  raise NotImplementedType()
    def getCrtFrame(self)-> Tuple[int, MatLike]:   raise NotImplementedType()
    def getCrtIndex(self)-> int: raise NotImplementedType()

    def reset(self) -> None: raise NotImplementedType()

    filters:List[Callable[[MatLike], MatLike]]|None = None

    def setFilters(self, filters:List[Callable[[MatLike], MatLike]]) -> None:
        self.filters = filters

    def _apply(self, frame:MatLike) -> MatLike:
        if self.filters is not None:
            for f in self.filters:
                frame = f(frame)
        return frame

    def __iter__(self) -> Self:
        self.reset()
        return self

    def __next__(self)-> Tuple[int, MatLike]:
        try:
            return self.getNextFrame()
        except IndexError:
            raise StopIteration
