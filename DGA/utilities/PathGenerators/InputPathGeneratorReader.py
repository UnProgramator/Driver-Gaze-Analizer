from collections.abc import Iterable

class InputPathGeneratorReader(Iterable[str]):
    def get_next_image_path(self) -> str: raise NotImplementedError
    def get_prev_image_path(self) -> str: raise NotImplementedError
    def get_crt_image_path(self) -> str: raise NotImplementedError
    def reset(self)->None :raise NotImplementedError

    def __next__(self):
        return self.get_next_image_path()

    def __iter__(self):
        self.reset()
        return self