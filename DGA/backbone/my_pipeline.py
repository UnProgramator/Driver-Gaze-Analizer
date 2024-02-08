import l2cs  #Pipeline
from l2cs.results import GazeResultContainer as l2cs_GazeResultContainer

import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


class my_pipeline():
    def __init__(self, 
            **kwargs
        ):
        self.pipeline = l2cs.Pipeline(**kwargs)

    def get_gaze(self, frame: np.ndarray, return_adnotate_frame:bool = False) -> (l2cs_GazeResultContainer):
        
        result = self.pipeline.step(frame)
        
        pitch = result.pitch
        yaw = result.yaw

        return result, pitch, yaw
