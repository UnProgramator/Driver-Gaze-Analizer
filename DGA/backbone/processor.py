from dataclasses import dataclass
from typing import Final
from typing_extensions import Self

from imageReaders.ImageReader import ImageReader
from .my_pipeline import my_pipeline
from l2cs import render
import cv2
import matplotlib.pyplot as plt
import csv
import os
import torch
import numpy as np


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
        '''
            @params:
                left, right, down, up - thresholds for looking in each direction
                minFrames - minimum duration in frame for filtration (is bellow, then the action is removed)
                splitThreshold - thrreshold for looking in a certain direction used to split the action words
        '''
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
        '''split the words into actions delimited by long face looking'''
        words=[]

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
        if len(words_f) > 0 and words_f[-1].direction == actions[i].direction:
            words_f[-1].noFrames += actions[i].noFrames
        else:
            words_f.append(actions[i].copy());
        


        self.action_array = words_f
        
        return self

    def process(self,imInput) -> Self:
        '''extracts the gaze from the images and saves it acordingly for furthure processing. generates the action list'''
        action_list = []
        for frame in imInput.images():
            results, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)

            if pitch.shape[0] != 1: # no face or more than one face detected
                continue;

            pitch:np.ndarray = pitch[0]
            yaw = yaw[0]
            
            cod = self.__codate(pitch, yaw)  #, pitch) # change after computing thresholds 
            action_list.append(cod)

        self.action_array = self.__reduce(action_list)
        self.words = None
        
        return self
    
    def codate_aparitions(self) -> list[list[int]] :
        '''generates the actions words'''
        cod_words:list[list[int]]=[]
        
        for word in self.words:
            cod_word:list[int]= self.__codate_template()
            for leter in word:
                char = leter.direction
                cod_word[self.__codate_template_map[char]]+=1
            cod_words.append(cod_word)
            
        self.last_code = cod_words
        return cod_words
    
    def codate_duration(self) ->list[list[int]] :
        '''codates the aparitions considering the percentage of durations foor looking in each direction'''
        cod_words:list[list[int]]=[]
        sum = 0;
        for word in self.words:
            cod_word:list[int]= self.__codate_template()
            for leter in word:
                char = leter.direction
                cod_word[self.__codate_template_map[char]]+=leter.noFrames
                sum+=leter.noFrames
            cod_words.append(cod_word)

        filterout = lambda x : list(map(lambda y: 100*y/sum, x))

        self.last_code = list(map(filterout, cod_words))
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
    def __codate(self,  pitch:int, yaw:int = 0):
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
        '''render the adnotated frames, mostly for validation or visual verification'''
        i=0;
        for frame,impath in imInput.images(True):
            results, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)
            if pitch is None or pitch is None:
                continue;
   
            frame = render(frame, results, color=(0, 255, 0))
            
            # Draw text annotations for original vectors
            cv2.putText(frame, "Original Pitch: " + str(pitch), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 0, 250), 1)
            cv2.putText(frame, "Original Yaw: " + str(yaw), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 0, 250), 1)

            # Draw text annotations for estimated vectors
            cv2.putText(frame, "System Pitch: " + str(results.pitch), (10, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 250, 0), 1)
            cv2.putText(frame, "System Yaw: " + str(results.yaw), (10, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 250, 0), 1)

            if savePath is not None:
                fullPath = savePath.format(imNr=i)
                i+=1
                cv2.imwrite(fullPath, frame)
                
            cv2.imshow("Demo", frame)

            key = cv2.waitKey(40)
            if key == 27:
                break
            

    def saveResults(self,imInput:ImageReader, savePath:str=None) -> None:
        '''plot the results, mostly for validation or visual verification'''
        pitches = []
        yaws = []
        results_pitches = []
        results_yaws = []

        for frame, impath in imInput.images(True):
            results, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)
            if pitch is None or pitch is None:
                continue;

            # Append values to lists
            pitches.append(pitch)
            yaws.append(yaw)
            results_pitches.append(results.pitch[0])
            results_yaws.append(results.yaw[0])

        # Save to CSV
        csv_filename = "gaze_estimations.csv"
        if savePath is not None:
            csv_filename = savePath

        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the rows for pitches and yaws
            csv_writer.writerow(['Original Pitch'] + pitches)
            csv_writer.writerow(['Original Yaw'] + yaws)
            csv_writer.writerow(['PF Pitch'] + results_pitches)
            csv_writer.writerow(['PF Yaw'] + results_yaws)
            
        return csv_filename

    def plot_gaze_estimations_from_csv(self, csv_filename):
        parent_dir = os.path.dirname(csv_filename)
        with open(csv_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = {rows[0]: rows[1:] for rows in csv_reader}

        # Convert string values to float
        pitches = list(map(float, data['Original Pitch']))
        yaws = list(map(float, data['Original Yaw']))
        results_pitches = list(map(float, data['System Pitch']))
        results_yaws = list(map(float, data['System Yaw']))

        frames = list(range(len(pitches)))

        # Plot pitch and results.pitch
        plt.figure(figsize=(10, 5))
        plt.plot(frames, pitches, label='Original Pitch', color='red')
        plt.plot(frames, results_pitches, label='System Pitch', color='green')
        plt.xlabel('Frame')
        plt.ylabel('Pitch')
        plt.title('Original Pitch vs. System Pitch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(parent_dir, 'pitch.png'))
        plt.show()

        # Plot yaw and results.yaw
        plt.figure(figsize=(10, 5))
        plt.plot(frames, yaws, label='Original Yaw', color='red')
        plt.plot(frames, results_yaws, label='System Yaw', color='green')
        plt.xlabel('Frame')
        plt.ylabel('Yaw')
        plt.title('Original Yaw vs. System Yaw')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(parent_dir, 'yaw.png'))
        plt.show()
            
    def validate(self, imInput) -> list[tuple[str, str,str,str]]:
        '''Forms an action list to be printed to the screen. For writing into a file, simply redirect the standard output'''
        action_list = []
        for frame, path in imInput.images(True):
            results, pitch, yaw = self.gaze_pipeline.get_gaze(frame, True)

            if pitch.shape[0] != 1: # no face or more than one face detected
                continue;

            pitch = pitch[0]
            yaw = yaw[0]
            
            gaze_dir_al:str = os.path.basename(path)
    
            idx = gaze_dir_al.index('.')
            gaze_dir = gaze_dir_al[idx-2:idx]
    
            idx = gaze_dir_al.index('_')
            driv = gaze_dir_al[idx+1:idx+3]

            cod = self.__codate(pitch, yaw)  #, pitch) # change after computing thresholds 
            action_list.append((path, gaze_dir, cod, driv))

        
        return action_list