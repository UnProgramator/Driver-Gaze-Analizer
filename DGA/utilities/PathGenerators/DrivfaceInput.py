from typing import Iterable
from typing_extensions import override
import cv2
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
            elif isinstance(iterSet, int):
                self.iterSet = (iterSet,)
            else:
                raise TypeError("Attribute iterSet was expected as int or tuple of ints, but {} was givven".format(type(iterSet)))
                
            #verify the elements in iterSet
            for i in self.iterSet:
                if i >= len(self.set_dim) or i<=0: #we selected an invalid set number
                    raise Exception("The colection contains only {} [0-{}] sets of drivers' pictures, but set with number {} was requested".format(len(self.set_dim), len(self.set_dim), i))      
        
    def __get_im_format(self,driver:int,nr:int,sufix:str='f '):
        if driver in [1,2]:
            nrs = "20130529"
        elif driver in [3,4]:
            nrs = "20130530"
        else:
            raise Exception()
        return "{0}_{1:02}_Driv_{2:03}_{3}.jpg".format(nrs,driver,nr,sufix)
        
    def __path(self):
        return os.environ['DATASETS']+ r'/drivface/DrivFace/DrivImages/DrivImages/'
        
    def __get_im(self,nr:int, driver:int = 1):
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
            return self.__get_im(self.last_im, self.im_set)
        else:
            raise IndexError()
    
    @override
    def images(self, returnPath:bool = False) -> Iterable:
        return DriverfaceInputImageIterator(self, returnPath)
        
    def __iter__(self): #set the curent set to the first iteration set
        self.iterCrt = 0
        self.change_set(self.iterSet[self.iterCrt]) 
        return self
    
    def __next__(self):
        if self.iterCrt is None:
            raise Exception("iterator not intialized or incorectly initialized")
        
        try:
            path = self.get_next_image_path()
            return path
        except IndexError:
            try:
                #paths in the curent set consumed, go to next set
                self.iterCrt+=1
                if self.iterCrt >= len(self.iterSet): # we consumed all elements
                    raise StopIteration
                # still have elements   
                self.change_set(self.iterSet[self.iterCrt])
                path = self.get_next_image_path()
                return path
            except:
                raise Exception("internal exception has occured")
    
    

class DriverfaceInputImageIterator(Iterable):
    dfIter:DrivfaceInput

    def __init__(self, drivfaceInput:DrivfaceInput, returnPath:bool):
        self.dfInput:DrivfaceInput = drivfaceInput
        self.returnPath = returnPath
        
    def __iter__(self):
        self.dfIter = iter(self.dfInput)
        return self
        
    def __next__(self):
        nextImgPath = next(self.dfIter)
        
        img = cv2.imread(nextImgPath)

        img = img[:,100:]#-100]

        if self.returnPath:
            return img, nextImgPath
        else:
            return img