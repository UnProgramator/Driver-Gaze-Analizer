import math
from typing import Tuple
import l2cs  #Pipeline
from l2cs.results import GazeResultContainer as l2cs_GazeResultContainer

import pathlib
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass

from backbone.particleFilter import ParticleFilter


class my_pipeline():
    def __init__(self, 
            **kwargs
        ):
        self.pipeline = l2cs.Pipeline(**kwargs)
        self.cntCorections = 0
        self.MAXCORECTIONS = 3  # Example value
        self.xtreshold = 0.2  # Example value
        
        self.particleFilterPitch = ParticleFilter(1000, np.empty((1,)))
        self.particleFilterYaw = ParticleFilter(1000, np.empty((1,)))

    def get_gaze(self, frame: np.ndarray, return_adnotate_frame:bool = False) -> Tuple[l2cs_GazeResultContainer, float, float]:
        """
        return the l2cs rezult containerm together with the pitch and yaw, int this order
        """
        result = self.pipeline.step(frame)
 
        frame = l2cs.render(frame, result)

        if not hasattr(self, 'pitch_system') or not hasattr(self, 'yaw_system'):
            self.pitch_system, self.yaw_system = self.get_pitch_yaw(result)


        raw_pitch, raw_yaw = self.get_pitch_yaw(result)
        if raw_pitch is None or raw_yaw is None:
            print("Warning: Raw Pitch or Raw Yaw is None")
            return result, None, None

        
        pitch_PF = self.particleFilterPitch.update(np.array([self.pitch_system]))
        yaw_PF = self.particleFilterYaw.update(np.array([self.yaw_system]))
        
        if abs(self.pitch_system - raw_pitch) > self.xtreshold or abs(self.yaw_system - raw_yaw) > self.xtreshold:
            self.cntCorections += 1
            if self.cntCorections < self.MAXCORECTIONS:
                # correction
                self.pitch_system = pitch_PF
                self.yaw_system = yaw_PF
                self.write_pitch_yaw(result, self.yaw_system, self.pitch_system)
            else:
                self.pitch_system = raw_pitch
                self.yaw_system = raw_yaw
                self.cntCorections = 0
        else:
            self.pitch_system = raw_pitch
            self.yaw_system = raw_yaw
            self.cntCorections = 0

        
        return result, raw_pitch, raw_yaw
    
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
        return result

