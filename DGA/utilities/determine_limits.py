from imageReaders.DrivfaceInput import DrivfaceInput
from backbone.my_pipeline import my_pipeline
import cv2
import os
import torch


imgsrc = DrivfaceInput((1,2,3,4))
gaze_pipeline = my_pipeline(
            weights=os.getcwd() +'/models/Gaze360/L2CSNet_gaze360.pkl',
            arch='ResNet50',
            device=torch.device('cpu')
        )

words = []

print('driv,yaw,pitch,gaze')

for frame, path in imgsrc.images(True):
    results, pitch, yaw = gaze_pipeline.get_gaze(frame, True)
            
    pitch = pitch[0]
    yaw = yaw[0]
         
    gaze_dir_al:str = os.path.basename(path)
    
    idx = gaze_dir_al.index('.')
    gaze_dir = gaze_dir_al[idx-2:idx]
    
    idx = gaze_dir_al.index('_')
    driv = gaze_dir_al[idx+1:idx+3]
    
    if gaze_dir[0]!='f':
        gaze_dir = gaze_dir[1]
    else:
        gaze_dir = gaze_dir[0]

    words.append((yaw,gaze_dir))
    print(driv, ',', yaw, ',' ,pitch,',' ,gaze_dir)