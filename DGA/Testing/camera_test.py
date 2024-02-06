from backbone.my_pipeline import my_pipeline
from l2cs import render
import cv2
import os
import torch

gaze_pipeline = my_pipeline(
    weights=os.getcwd() +'/models/Gaze360/L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu')
)

print("pipeline created succesfully")
 
cap = cv2.VideoCapture(0)

print("camera capture initialized fuccesfully")

while True:
    _, frame = cap.read()    

    # Process frame and visualize
    results, pitch, yaw = gaze_pipeline.get_gaze(frame, True)
    
    pitch = pitch[0]
    yaw = yaw[0]
    
    frame = render(frame, results)
    
    cv2.putText(frame, "Pitch: " + str(pitch), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "  Yaw: " + str(yaw), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
    cv2.imshow("Demo", frame)
    
    if cv2.waitKey(1) == 27:
        break