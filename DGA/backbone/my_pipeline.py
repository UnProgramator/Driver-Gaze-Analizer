import math
import l2cs  #Pipeline
from l2cs.results import GazeResultContainer as l2cs_GazeResultContainer

import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from backbone.particleFilter import GazeEstimator


class my_pipeline():
    def __init__(self, 
            **kwargs
        ):
        self.pipeline = l2cs.Pipeline(**kwargs)

    def get_gaze(self, frame: np.ndarray, return_adnotate_frame:bool = False) -> (l2cs_GazeResultContainer):
        
        result = self.pipeline.step(frame)
        
        pitch = result.pitch
        yaw = result.yaw
        frame = l2cs.render(frame, result)
        
        # fir extragi de genul yaw = get_one_face(yaw) et so on
        # sau ambele folosind get_pitch_yaw
        # calculezi noile valori
        # inainte sa le returnezi sa le si scrii inapoi in results, altfel nu o sa le randeze bine
        # poti folosi write_pitch_yaw(...)
        raw_pitch, raw_yaw = self.get_pitch_yaw(result)
        
        gaze_estimator = GazeEstimator()
        estimated_pitch, estimated_yaw = gaze_estimator.filter_gaze(raw_pitch, raw_yaw)
        self.write_pitch_yaw(result, estimated_yaw, estimated_pitch)

        return result, pitch, yaw
    
    def get_one_face(self, dir:np.ndarray):
        #if dir.shape[0] != 1:
        #    raise Exception("More than one face detected")
        if dir.shape[0] == 1:
            return dir[0]
    
    def get_pitch_yaw(self, result):
        return self.get_one_face(result.pitch), self.get_one_face(result.yaw)
    
    def write_pitch_yaw(self, result, new_yaw, new_pitch):
        result.pitch[0] = new_pitch
        result.yaw[0]   = new_yaw
        
    def unit_vector_coordinates(gaze_angle):
        theta_deg = gaze_angle * 180
        # Convert degrees to radians
        theta_rad = math.radians(theta_deg)
    
        # Calculate coordinates of unit vector
        x = math.cos(theta_rad)
        y = math.sin(theta_rad)
    
        return x, y
