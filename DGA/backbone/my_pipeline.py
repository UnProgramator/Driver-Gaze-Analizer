import l2cs  #Pipeline
from l2cs.results import GazeResultContainer as l2cs_GazeResultContainer

from typing import Any, Tuple

import cv2
import numpy as np


class my_pipeline():

    lastresult:l2cs_GazeResultContainer|None = None

    def __init__(self, 
            **kwargs
        ):
        self.pipeline = l2cs.Pipeline(**kwargs)

    def get_gaze(self, frame: cv2.typing.MatLike, return_adnotate_frame:bool = False) \
                -> Tuple[l2cs_GazeResultContainer, \
                         np.ndarray[Any, np.dtype[np.float32]],\
                         np.ndarray[Any, np.dtype[np.float32]]]:
        
        try:
            result = self.pipeline.step(frame)
            
        
            pitch:np.ndarray[Any, np.dtype[np.float32]] = result.pitch
            yaw:np.ndarray[Any, np.dtype[np.float32]] = result.yaw
        

            # fir extragi de genul yaw = get_one_face(yaw) et so on
            # sau ambele folosind get_pitch_yaw
            # calculezi noile valori
            # inainte sa le returnezi sa le si scrii inapoi in results, altfel nu o sa le randeze bine
            # poti folosi write_pitch_yaw(...)

            if pitch.shape[0] != 1:
                if self.lastresult is not None:
                    i = self.get_min_dist_idx(result)
                    pitch = pitch[i:i+1]
                    yaw = yaw[i:i+1]
                    assert(pitch.shape[0] == 1)
                    result.pitch = pitch
                    result.yaw = yaw
                    result.bboxes = result.bboxes[i:i+1]
                    result.landmarks = result.landmarks[i:i+1]
                    result.scores = result.scores[i:i+1]
                else:
                    raise FirstFrameNoFaceException()


            self.lastresult = result
            return result, pitch, yaw
        except ValueError as e:
            print(e)
            raise NoFaceException(str(e))
    
    def get_one_face(self, src:np.ndarray[Any, np.dtype[np.float32]]) -> float:
        if src.shape[0] != 1:
            raise Exception("More than one face detected")
        
        return src[0]
    
    def get_pitch_yaw(self, result:l2cs_GazeResultContainer):
        return self.get_one_face(result.pitch), self.get_one_face(result.yaw)
    
    def write_pitch_yaw(self, result:l2cs_GazeResultContainer, new_yaw:float, new_pitch:float)\
                -> l2cs_GazeResultContainer:
        result.pitch[0] = new_pitch
        result.yaw[0]   = new_yaw
        return result

    def get_min_dist_idx(self, results:l2cs_GazeResultContainer) -> int:
        '''
            In case there are multiple bounding boxes, verify which is closer to the last detected bounding box
        '''
        if self.lastresult is None:
            raise ValueError(f'variable lastresult, class {type(self)} is not initialized before is used')
        last_bb:np.ndarray[Any, np.dtype[np.float32]]= (self.lastresult.bboxes[0][0:2] + self.lastresult.bboxes[0][2:4])/2

        dmin:float = -1.
        idx = 0

        for i in range(results.pitch.shape[0]):
            box:np.ndarray[Any, np.dtype[np.float32]] = (results.bboxes[i][0:2] + results.bboxes[i][2:4])/2
            d:float = np.linalg.norm(box - last_bb)
            if dmin == -1.:
                dmin = d
            elif dmin > d:
                dmin = d
                idx = i

        return idx
    


class NoFaceException(Exception):
    pass

class FirstFrameNoFaceException(Exception):
    pass