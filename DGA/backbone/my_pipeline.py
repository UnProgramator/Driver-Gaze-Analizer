import l2cs  #Pipeline
from l2cs.results import GazeResultContainer as l2cs_GazeResultContainer

from typing import Any, Tuple

import cv2
import numpy as np


class my_pipeline():
    def __init__(self, 
            **kwargs
        ):
        self.pipeline = l2cs.Pipeline(**kwargs)

    def get_gaze(self, frame: cv2.typing.MatLike, return_adnotate_frame:bool = False) -> Tuple[l2cs_GazeResultContainer, np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
        
        result = self.pipeline.step(frame)
        
        pitch:np.ndarray[Any, np.dtype[np.float32]] = result.pitch
        yaw:np.ndarray[Any, np.dtype[np.float32]] = result.yaw
        

        # fir extragi de genul yaw = get_one_face(yaw) et so on
        # sau ambele folosind get_pitch_yaw
        # calculezi noile valori
        # inainte sa le returnezi sa le si scrii inapoi in results, altfel nu o sa le randeze bine
        # poti folosi write_pitch_yaw(...)

        return result, pitch, yaw
    
    def get_one_face(self, src:np.ndarray[Any, np.dtype[np.float32]]) -> float:
        if src.shape[0] != 1:
            raise Exception("More than one face detected")
        
        return src[0]
    
    def get_pitch_yaw(self, result:l2cs_GazeResultContainer):
        return self.get_one_face(result.pitch), self.get_one_face(result.yaw)
    
    def write_pitch_yaw(self, result:l2cs_GazeResultContainer, new_yaw:float, new_pitch:float) -> l2cs_GazeResultContainer:
        result.pitch[0] = new_pitch
        result.yaw[0]   = new_yaw
        return result
    
