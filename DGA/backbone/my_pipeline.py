import l2cs  #Pipeline
from l2cs.results import GazeResultContainer as l2cs_GazeResultContainer

import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


class my_pipeline(l2cs.Pipeline):
    def __init__(this, 
            **kwargs
        ):
        super().__init__(**kwargs)

    def get_gaze(self, frame: np.ndarray, return_adnotate_frame:bool = False):
        
        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)
            
            #print("faces", len(faces))
            
            if faces is not None: 
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < self.confidence_threshold:
                        continue

                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            else:

                pitch = np.empty((0,1))
                yaw = np.empty((0,1))

        else:
            pitch, yaw = self.predict_gaze(frame)
        
        
        if return_adnotate_frame:
            # Save data
            results = l2cs_GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.stack(bboxes),
                landmarks=np.stack(landmarks),
                scores=np.stack(scores)
            )

        return results, pitch, yaw
