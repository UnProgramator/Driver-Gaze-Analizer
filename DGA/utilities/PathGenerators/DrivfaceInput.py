from typing_extensions import override
import os
from .InputPathGeneratorReader import InputPathGeneratorReader

#20130529_01_Driv_001_f  20130529_01_Driv_179_ll
#20130529_02_Driv_001_f  20130529_02_Driv_170_f 
#20130530_03_Driv_001_f  20130530_03_Driv_167_f 
#20130530_04_Driv_001_f  20130530_04_Driv_090_f 

class DrivfaceInput(InputPathGeneratorReader):
    set_dim = (0, 179, 170, 167, 90) # the driver number ranges from 1 to 4. 0 is put for convenience only

    def __init__(self, iterSet:tuple[int]|int|None=None):
        self.change_set(1)
        self.iterCrt = None
        
        if iterSet is None:
            self.iterSet = (1,2,3,4)
        else:
            if isinstance(iterSet, tuple):
                self.iterSet =  iterSet
            else:
                self.iterSet = (iterSet,)
                
            #verify the elements in iterSet
            for i in self.iterSet:
                if i >= len(self.set_dim) or i<=0: #we selected an invalid set number
                    raise Exception("The colection contains only {} [0-{}] sets of drivers' pictures, but set with number {} was requested".format(len(self.set_dim), len(self.set_dim), i))      
        
    def __get_im_format(self,driver:int,nr:int,sufix:str='f ') -> str:
        if driver in [1,2]:
            nrs = "20130529"
        elif driver in [3,4]:
            nrs = "20130530"
        else:
            raise Exception()
        return "{0}_{1:02}_Driv_{2:03}_{3}.jpg".format(nrs,driver,nr,sufix)
        
    def __path(self) -> str:
        return os.environ['DATASETS']+ r'/drivface/DrivFace/DrivImages/DrivImages/'
        
    def __get_im(self,nr:int, driver:int = 1) -> str:
        pat = self.__path() + self.__get_im_format(driver, nr)
        if os.path.exists(pat):
            return pat
            
        pat = self.__path() + self.__get_im_format(driver, nr, 'lr')
        if os.path.exists(pat):
            return pat
            
        pat = self.__path() + self.__get_im_format(driver, nr, 'll')
        if os.path.exists(pat):
            return pat

        raise Exception(f'bad arguments for function __get_im')
            
    
    def change_set(self, new_im_set:int) -> None:
        if not( 1 <= new_im_set <=4):
            raise Exception('new image set number ({}) is out of valid values ([1,4])'.format(new_im_set))
            
        self.im_set = new_im_set
        self.last_im = 0
        self.set_size = self.set_dim[self.im_set]
    
    @override
    def get_next_image_path(self) -> str:
        try:
            if self.last_im >= self.set_size:
                self.change_set(self.im_set+1)
            self.last_im+=1
            return self.__get_im(self.last_im, self.im_set)
        except:
            raise IndexError()
    
    @override
    def get_prev_image_path(self) -> str:
        try:
            if self.last_im <= 0:
                self.change_set(self.im_set+1)
            self.last_im+=1
            return self.__get_im(self.last_im, self.im_set)
        except:
            raise IndexError()

    @override
    def get_crt_image_path(self) -> str:
        return self.__get_im(self.last_im, self.im_set)

    @override
    def reset(self)->None :
        self.change_set(0)
    
    