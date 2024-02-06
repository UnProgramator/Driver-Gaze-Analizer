from dataclasses import dataclass
from typing import Final
from typing_extensions import Self

from imageReaders.ImageReader import ImageReader
from .my_pipeline import my_pipeline
from l2cs import render
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
    
    def __codate_template(_) -> list[int] : return [0, 0, 0, 0, 0,].copy()
    __codate_template_map:Final[dict[str,int]] = {'f':0, 'l':1, 'r':2, 'u':3, 'd':4}

    def __init__(self,  left:int=0, right:int=0, down:int=0, up:int=0, minFames:int=1, splitThreshold:int=40):
        self.left:Final[int] = left
        self.right:Final[int]  = right 
        self.down:Final[int] = down
        self.up:Final[int] = up
        
        self.min_frames_no:Final[int] = minFames
        
        self.split_th:Final[int] = splitThreshold

        self.action_array:list[WordsEntry] = None        
        self.words:list[list[WordsEntry]] = None
        
        self.last_code:list[list[int]] = None

        self.gaze_pipeline = my_pipeline(
            weights=os.getcwd() +'/models/Gaze360/L2CSNet_gaze360.pkl',
            arch='ResNet50',
            device=torch.device('cpu')
        )

    def get_words(self) ->list[list[WordsEntry]]:
        if self.words is None:
            raise Exception("words not split yet")
        return self.words
    
    def get_action_list(self) ->list[WordsEntry]:
        if self.action_array is None:
            raise Exception("words not split yet")
        return self.action_array

    def split_words(self) -> Self:
        #split the words into actions delimited by long face looking
        words=[]

        s:int = 0
        f:int = 0
        ar = self.action_array

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

        #if a string of actions ends at the end of the action list
        if s != f : # guard to not introduce a delimiter found as the last action
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
        
        words_f:list[WordsEntry] =[]

        # for word in actions:
        #     if word.noFrames < self.min_frames_no and  len(words_f) and words_f[-1].direction == word.direction:
        #         words_f[-1].noFrames += word.noFrames
        #     else:
        #         words_f.append(word)

        actions = self.action_array

        for i in range(1, len(actions)-1):
            if actions[i].noFrames < self.min_frames_no and actions[i-1].direction == actions[i+1].direction: 
                #ignore frame if action is to short and the frames that border it are the same
                if len(words_f):
                    words_f[-1].noFrames += actions[i].noFrames #if turbulent frame, consider frame to be the frame that borders it
                continue; # if turbulent frame, then ignore
            if len(words_f) and words_f[-1].direction == actions[i].direction:
                words_f[-1].noFrames+=actions[i].noFrames
            else:
                words_f.append(actions[i].copy());
        
        


        self.action_array = words_f
        
        return self

    def process(self,imInput) -> Self:
        action_list = []
        for frame in imInput.images():
            results, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)
            
            pitch = pitch[0]
            yaw = yaw[0]
            
            yaw = self.__codate(yaw)  #, pitch) # change after computing thresholds 
            action_list.append(yaw)
        
        self.action_array = self.__reduce(action_list)
        self.words = None
        
        return self
    
    def codate_aparitions(self) -> list[list[int]]  :return self.codate_words(0)
    
    def codate_duration(self) ->list[list[int]]     :return self.codate_words(1)

    def codate_words(self, cod_type:int=0) -> list[list[int]]:
        
        cod_words:list[list[int]]=[]
        
        for word in self.words:
            cod_word:list[int]= self.__codate_template()
            for leter in word:
                char = leter.direction
                if cod_type ==0:
                    cod_word[self.__codate_template_map[char]]+=1
                elif cod_type ==1:
                    cod_word[self.__codate_template_map[char]]+=leter.noFrames
                else:
                    raise Exception(f'invalid argument cod_type = {cod_type}')
            cod_words.append(cod_word)
            
        self.last_code = cod_words
        return cod_words
    
    # def decode(self, code):
    #     pass

    ##########################################################################
    def __codate(self,  pitch:int, yaw:int = 0):
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
        crt:str = None
        nr:int = None
        result:list[WordsEntry] = []
        for i in ar:
            if i == crt:
                nr+=1
            else:
                if crt is not None:
                    result.append(WordsEntry(crt, nr))
                crt = i
                nr = 1
        result.append(WordsEntry(crt, nr))
        return result


    ##########################################################################
    def render(self,imInput:ImageReader, savePath:str=None) -> None:
        i=0;
        for frame,impath in imInput.images(True):
            results, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)
            
            pitch = pitch[0]
            yaw = yaw[0]

                
            frame = render(frame, results)
            
            cv2.putText(frame, os.path.basename(impath), (0,95), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 250), 2)
            cv2.putText(frame, "Pitch: " + str(pitch), (0, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 0, 250), 1)
            cv2.putText(frame, "  Yaw: " + str(yaw), (0, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 0, 250), 1)
            cv2.putText(frame, "  Dir1: " + str(self.__codate(pitch)), (0, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 0, 250), 1)
            cv2.putText(frame, "  Dir2: " + str(self.__codate(pitch,yaw)), (0, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 0, 250), 1)

            if savePath is not None:
                fullPath = savePath.format(imNr=i)
                i+=1
                cv2.imwrite(fullPath, frame)
                
            cv2.imshow("Demo", frame)
            

            key = cv2.waitKey(40)
            
            

            if key == 27:
                break
            
