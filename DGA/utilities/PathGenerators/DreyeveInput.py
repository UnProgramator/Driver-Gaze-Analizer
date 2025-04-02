from typing import Final, Tuple
from typing_extensions import deprecated, override
import os
from .InputPathGeneratorReader import InputPathGeneratorReader

@deprecated
class DrivfaceInput(InputPathGeneratorReader):
    set_size:int
    set_dim:Final[Tuple[int]] = (0,)

    def __init__(self):
        self.change_set(1)
        self.iterCrt = None          
        
    def __get_im_format(self,driver:int,nr:int,sufix:str='f ') -> str:
        if driver in [1,2]:
            nrs = "20130529"
        elif driver in [3,4]:
            nrs = "20130530"
        else:
            raise Exception()
        return "{0}_{1:02}_Driv_{2:03}_{3}.jpg".format(nrs,driver,nr,sufix)
        
    def __path(self) -> str:
        return r'D:\Programming\VisualStudio\Python\Driver-Gaze-Analizer\DGA\dataset\drivface\DrivFace\DrivImages\\'
        
    def __get_im_pat(self,nr:int, driver:int = 1)-> str:
        pat = self.__path() + self.__get_im_format(driver, nr)
        if os.path.exists(pat):
            return pat
            
        pat = self.__path() + self.__get_im_format(driver, nr, 'lr')
        if os.path.exists(pat):
            return pat
            
        pat = self.__path() + self.__get_im_format(driver, nr, 'll')
        if os.path.exists(pat):
            return pat

        raise Exception()
            
    
    def change_set(self, new_im_set:int):
        if not( 1 <= new_im_set <=4):
            raise Exception('new image set number ({}) is out of valid values ([1,4])'.format(new_im_set))
            
        self.im_set = new_im_set
        self.last_im = 0
        self.set_size = self.set_dim[self.im_set]
    
    @override
    def get_next_image_path(self) -> str:
        if self.last_im < self.set_size:
            self.last_im+=1
            return self.__get_im_pat(self.last_im, self.im_set)
        else:
            raise IndexError()

    @override
    def get_prev_image_path(self) -> str:
        if self.last_im > 0:
            self.last_im-=1
            return self.__get_im_pat(self.last_im, self.im_set)
        else:
            raise IndexError()
    
    @override
    def get_crt_image_path(self) -> str:
        return self.__get_im_pat(self.last_im, self.im_set)

    @override
    def reset(self)->None :
        pass