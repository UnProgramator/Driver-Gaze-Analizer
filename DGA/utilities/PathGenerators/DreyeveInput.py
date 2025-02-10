from typing_extensions import override
import cv2
import os
from .InputPathGeneratorReader import InputPathGeneratorReader


class DrivfaceInput(InputPathGeneratorReader):

    def __init__(self):
        self.change_set(1)
        self.iterCrt = None
        
            
        
    def __get_im_format(self,driver:int,nr:int,sufix:str='f '):
        if driver in [1,2]:
            nrs = "20130529"
        elif driver in [3,4]:
            nrs = "20130530"
        else:
            raise Exception()
        return "{0}_{1:02}_Driv_{2:03}_{3}.jpg".format(nrs,driver,nr,sufix)
        
    def __path(self):
        return r'D:\Programming\VisualStudio\Python\Driver-Gaze-Analizer\DGA\dataset\drivface\DrivFace\DrivImages\\'
        
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
            
    
    def change_set(self, new_im_set:int):
        if not( 1 <= new_im_set <=4):
            raise Exception('new image set number ({}) is out of valid values ([1,4])'.format(new_im_set))
            
        self.im_set = new_im_set
        self.last_im = 0
        self.set_size = self.set_dim[self.im_set]
    
    @override
    def get_next_image_path(self):
        if self.last_im < self.set_size:
            self.last_im+=1
            return self.__get_im(self.last_im, self.im_set)
        else:
            return None
    
    @override
    def images(self, returnPath:bool = False):
        return DriverfaceInputImageIterator(self, returnPath)
        
    def __iter__(self): #set the curent set to the first iteration set
        self.iterCrt = 0
        self.change_set(self.iterSet[self.iterCrt]) 
        return self
    
    def __next__(self):
        if self.iterCrt is None:
            raise Exception("iterator not intialized or incorectly initialized")
            
        path = self.get_next_image_path()
        if path is not None: # valid path
            
            return path
        #paths in the curent set consumed, go to next set
        self.iterCrt+=1
        if self.iterCrt >= len(self.iterSet): # we consumed all elements
            raise StopIteration
        # still have elements   
        self.change_set(self.iterSet[self.iterCrt])
        path = self.get_next_image_path()
        if path is not None: # valid path
            return path
        raise Exception("internal exception has occured")
    
    

class DriverfaceInputImageIterator:
    def __init__(self, drivfaceInput, returnPath:bool):
        self.dfInput = drivfaceInput
        self.dfIter = None
        self.returnPath = returnPath
        
    def __iter__(self):
        self.dfIter = iter(self.dfInput)
        return self
        
    def __next__(self):
        nextImgPath = next(self.dfIter)
        if nextImgPath is None:
            raise StopIteration
        
        img = cv2.imread(nextImgPath)

        img = img[:,100:]#-100]

        if self.returnPath:
            return img, nextImgPath
        else:
            return img