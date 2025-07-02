from typing import Callable, Iterator, List, Tuple
from types import NotImplementedType
from cv2.typing import MatLike

class IReader(Iterator[Tuple[int, MatLike]]):
    def getNextFrame(self) -> Tuple[int, MatLike]: raise NotImplementedType()
    def getPrevFrame(self)-> Tuple[int, MatLike]:  raise NotImplementedType()
    def getCrtFrame(self)-> Tuple[int, MatLike]:   raise NotImplementedType()
    def getCrtIndex(self)-> int: raise NotImplementedType()

    def reset(self) -> None: raise NotImplementedType()

    filters:List[Callable[[MatLike], MatLike]]|None = None

    def setFilters(self, filters:List[Callable[[MatLike], MatLike]]) -> None:
        self.filters = filters

    def addFilters(self, filters:List[Callable[[MatLike], MatLike]]|Callable[[MatLike], MatLike]) -> None:
        if self.filters is not None:
            self.filters += filters if isinstance(filters,List) else [filters]
        else:
            self.filters = filters if isinstance(filters,List) else [filters]

    def _apply(self, frame:MatLike) -> MatLike:
        if self.filters is not None:
            for f in self.filters:
                frame = f(frame)
        return frame

    def __iter__(self):
        self.reset()
        return self

    def __next__(self)-> Tuple[int, MatLike]:
        try:
            return self.getNextFrame()
        except IndexError:
            raise StopIteration
