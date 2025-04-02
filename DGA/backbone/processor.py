from dataclasses import dataclass
from typing import Any, Callable, Final, List, Tuple
from cv2.typing import MatLike
from typing_extensions import Self

from utilities.ImageReaders.IReader import IReader
from utilities.PathGenerators.InputPathGeneratorReader import InputPathGeneratorReader
from .my_pipeline import my_pipeline
from l2cs import render
import numpy as np
import cv2
import os
import torch


@dataclass
class WordsEntry:
    direction:str
    noFrames:int
    
    def copy(self):
        return WordsEntry(self.direction, self.noFrames)
        

class Processor:
    
    @staticmethod
    def __codate_template() -> list[int] : return [0, 0, 0, 0, 0,].copy()
    __codate_template_map:Final[dict[str,int]] = {'f':0, 'l':1, 'r':2, 'u':3, 'd':4} 
    
    action_array:list[WordsEntry]|None = None
    words:list[list[WordsEntry]]|None = None   
    last_code:list[list[int]]|None = None
    gaze_pipeline:my_pipeline

    def __init__(self,  left:float=0, right:float=0, down:float=0, up:float=0, minFames:int=1, splitThreshold:int=40):
        '''
            @params:
                left, right, down, up - thresholds for looking in each direction
                minFrames - minimum duration in frame for filtration (is bellow, then the action is removed)
                splitThreshold - thrreshold for looking in a certain direction used to split the action words
        '''
        self.left:Final[float] = left
        self.right:Final[float]  = right 
        self.down:Final[float] = down
        self.up:Final[float] = up
        
        self.min_frames_no:Final[int] = minFames
        
        self.split_th:Final[int] = splitThreshold

        
        try:
            self.gaze_pipeline = my_pipeline(
                weights=os.environ['L2CS_weights'] +'Gaze360/L2CSNet_gaze360.pkl',
                arch='ResNet50',
                device=torch.device('cpu')
            )
        except KeyError:
            print('L2CS_eights envrionemtnal variable not set. please set as the directory containing the weights files (eg. Gaze360/L2CSNet_gaze360.pkl)' )
            exit(100)

    def get_words(self) ->list[list[WordsEntry]]:
        if self.words is None:
            raise Exception("words not split yet")
        return self.words
    
    def get_action_list(self) ->list[WordsEntry]:
        if self.action_array is None:
            raise Exception("words not split yet")
        return self.action_array

    def split_words(self) -> Self:
        '''split the words into actions delimited by long face looking'''
        words:list[list[WordsEntry]]=[]

        if self.action_array is None:
            raise Exception()

        ar = self.action_array
        s:int = 0
        f:int = 0
        

        for i in range(len(ar)):
            if ar[i].noFrames < self.split_th or ar[i].direction != 'f': #crt action time is in the treshold
                continue
            f = i
            
            if s != f : # guard to not introduce a delimiter found as the first action
                word:list[WordsEntry] = [act.copy() for act in ar[s:f]]
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #formula ar trebui revizuita!
                # if i!=0:
                #     word[0].noFrames = min(word[0].noFrames, self.split_th)
                # if i!=len(ar):
                #     word[-1].noFrames = min(word[-1].noFrames, self.split_th)

                words.append(word)
            
            s = i+1 #the start of a new word is the next action after the pause after the curent word
        
        f:int = len(ar)-1
        
        #if a string of actions ends at the end of the action list
        if s != f and s<f : # guard to not introduce a delimiter found as the last action
                word:list[WordsEntry] = [act.copy() for act in ar[s:f]]
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #formula ar trebui revizuita!
                # if i!=0:
                #     word[0].noFrames = min(word[0].noFrames, self.split_th)
                # if i!=len(ar):
                #     word[-1].noFrames = min(word[-1].noFrames, self.split_th)

                words.append(word)

        self.words = words
        return self

    def reduce_noise(self) -> Self:
        '''remove actions that which lasted less than a threshold'''
        words_f:list[WordsEntry] =[]

        # for word in actions:
        #     if word.noFrames < self.min_frames_no and  len(words_f) and words_f[-1].direction == word.direction:
        #         words_f[-1].noFrames += word.noFrames
        #     else:
        #         words_f.append(word)

        if self.action_array is None:
            raise Exception()

        actions = self.action_array

        for i in range(1, len(actions)-1):
            
            if actions[i].noFrames < self.min_frames_no:# and actions[i-1].direction == actions[i+1].direction: 
                #ignore frame if action is to short and the frames that border it are the same
                continue; # if turbulent frame, then ignore
            
            if len(words_f) > 0 and words_f[-1].direction == actions[i].direction:
                words_f[-1].noFrames += actions[i].noFrames
            else:
                words_f.append(actions[i].copy());
        
        #for the last word
        i = len(actions)-1
        if len(words_f) > 0 and words_f[-1].direction == actions[i].direction:
            words_f[-1].noFrames += actions[i].noFrames
        else:
            words_f.append(actions[i].copy());
        


        self.action_array = words_f
        
        return self

    def process(self,imInput:IReader) -> Self:
        '''extracts the gaze from the images and saaves it acordingly for furthure processing. generates the action list'''
        action_list:list[str] = []
        for _,frame in imInput:
            pitch:np.ndarray[Any, np.dtype[np.float32]]
            yaw:np.ndarray[Any, np.dtype[np.float32]]
            _,pitch,yaw= self.gaze_pipeline.get_gaze(frame, True) # type: ignore

            if pitch.shape[0] != 1: # no face or more than one face detected
                continue;

            _pitch:float = pitch[0]
            _yaw:float = yaw[0]
            
            cod = self.__codate(_pitch, _yaw)  #, pitch) # change after computing thresholds 
            action_list.append(cod)

        self.action_array = self.__reduce(action_list)
        self.words = None
        
        return self
    
    def codate_aparitions(self, add_tt:bool=False) -> list[list[int]] :
        '''codates the number of transitions to each general direction of gaze, and concat the total time at the end if requested'''
        cod_words:list[list[int]]=[]
        
        if self.words is None:
            raise Exception()

        for word in self.words:
            sum_t = 0
            cod_word:list[int]= self.__codate_template()
            for leter in word:
                char = leter.direction
                if add_tt: sum_t += leter.noFrames
                cod_word[self.__codate_template_map[char]]+=1
            if add_tt: cod_word.append(sum_t) #add total time, if requested
            cod_words.append(cod_word)
            
        self.last_code = cod_words
        return cod_words
    
    #original
    # def codate_duration(self) ->list[list[int]] :
    #     '''codates the aparitions considering the percentage of durations foor looking in each direction'''
    #     cod_words:list[list[int]]=[]
    #     sum = 0;
    #     for word in self.words:
    #         cod_word:list[int]= self.__codate_template()
    #         for leter in word:
    #             char = leter.direction
    #             cod_word[self.__codate_template_map[char]]+=leter.noFrames
    #             sum+=leter.noFrames
    #         cod_words.append(cod_word)

    #     filterout = lambda x : list(map(lambda y: 100*y/sum, x))

    #     self.last_code = list(map(filterout, cod_words))
    #     return self.last_code

    def codate_duration_tt(self, add_tt:bool=False) ->list[list[int]] :
        '''codates the aparitions considering the percentage of durations foor looking in each direction, and concat the total time at the end if requested'''

        if self.words is None:
            raise Exception()

        cod_words:list[list[int]]=[]
        filterout:Callable[[int], Callable[[int],int]] = lambda t : ( lambda x : int(100*x/t) )

        for word in self.words:
            cod_word:list[int]= self.__codate_template()
            sum_t = 0
            for leter in word:
                char = leter.direction
                cod_word[self.__codate_template_map[char]]+=leter.noFrames
                sum_t+=leter.noFrames
            cod_word = list(map(filterout(sum_t), cod_word))
            if add_tt: cod_word.append(sum_t) #add total time, if requested
            cod_words.append(cod_word)
        
        self.last_code = cod_words
        
        return self.last_code
    
    




    # def decode(self, code):
    #     pass

    def load_array(self, action_array:list[tuple[int, int]]):
        '''Directly input the number array'''
        act_ar:list[str] = []
        for X,Y in action_array:
            act_ar.append(self.__codate(X,Y))
                          
        self.action_array = self.__reduce(act_ar)
        self.words = None
        
        return self
        

    ##########################################################################
    def __codate(self,  pitch:float, yaw:float = 0):
        '''codate the direction using a character, for conviniecne'''
        if yaw < self.down:
            return 'd'
        if yaw > self.up:
            return 'u'

        if pitch < self.left:
            return 'l'
        if pitch > self.right:
            return 'r'
        
        return 'f'
    
    def __reduce(self, ar:list[str]):
        '''reduce the gaze list to the action list?'''
        crt:str|None = None
        nr:int = 0
        result:list[WordsEntry] = []
        for i in ar:
            if i == crt:
                nr+=1
            else:
                if crt is not None:
                    result.append(WordsEntry(crt, nr))
                crt = i
                nr = 1
        if crt is None:
            raise Exception
        result.append(WordsEntry(crt, nr))
        return result


    ##########################################################################
    def render(self,imInput:IReader, savePath:str|None=None, start_idx:int=0) -> int:
        '''render the adnotated frames, mostly for validation or visual verification, returns the last used index'''
        for _ , frame in imInput:

            pitch_a:np.ndarray[Any, np.dtype[np.float32]]
            yaw_a:np.ndarray[Any, np.dtype[np.float32]]
            results, pitch_a, yaw_a = self.gaze_pipeline.get_gaze(frame, True)
            
            pitch:float = pitch_a[0]
            yaw:float = yaw_a[0]

                
            frame:MatLike = render(frame, results)
           
            cv2.putText(frame, "        Pitch: " + str(pitch), (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 0, 250), 1)
            cv2.putText(frame, "         Yaw: " + str(yaw), (0, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 0, 250), 1)
            cv2.putText(frame, "Predicted direction: " + str(self.__codate(pitch,yaw)), (0, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 0, 250), 1)

            if savePath is not None:
                fullPath = savePath.format(imNr=(start_idx+start_idx))
                start_idx+=1
                cv2.imwrite(fullPath, frame)
                
            cv2.imshow("Demo", frame)

            key = cv2.waitKey(40)

            if key == 27:
                break
        return start_idx
            

    def validate(self, imInput:IReader, pat:InputPathGeneratorReader) -> list[tuple[str, str,str,str]]:
        '''Forms an action list to be printed to the screen. For writing into a file, simply redirect the standard output'''
        action_list:List[Tuple[str,str,str,str]] = []
        for _, frame in imInput:
            _, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)

            if pitch.shape[0] != 1: # no face or more than one face detected
                continue;

            pitch = pitch[0]
            yaw = yaw[0]
            path = pat.get_crt_image_path()
            
            gaze_dir_al:str = os.path.basename(path)
    
            idx = gaze_dir_al.index('.')
            gaze_dir = gaze_dir_al[idx-2:idx]
    
            idx = gaze_dir_al.index('_')
            driv = gaze_dir_al[idx+1:idx+3]

            cod = self.__codate(pitch, yaw)  #, pitch) # change after computing thresholds 
            action_list.append((path, gaze_dir, cod, driv))

        
        return action_list


    def get_pitch_yaw_list(self, imInput:IReader) -> list[tuple[int, float, float]]:
        '''get an IReader or an InputPathGeneratorReader'''
        res:list[tuple[int, float, float]] = []
        for frid, frame in imInput:
            _, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)
            pitch = pitch[0]
            yaw = yaw[0]

            res += [(frid, pitch, yaw)]

        return res