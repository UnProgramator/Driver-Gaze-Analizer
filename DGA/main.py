from calendar import c
from multiprocessing import process

from sklearn.cluster import KMeans
from imageReaders.DrivfaceInput import DrivfaceInput
from backbone.clustering import clustering
from backbone.my_pipeline import my_pipeline
import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from backbone.processor import Processor

from utilities.Validation.dreyeve_validation import drvalidation


def gproc():
    return Processor(-0.65, 0.2, -0.5, 0.8, 2, 30 )

def f1(imgsrc):
    

    proc = gproc()

    cod = proc.process(imgsrc).split_words().codate_aparitions()
    
    
    print(*proc.get_action_list())
    print()
    for w in proc.get_words(): print(w)
    print()
    print(*cod)

def f4(imgsrc):
    proc = gproc()
    sum = 0
    ar = proc.validate(imgsrc)
    for pair in ar:
        print(pair[0],',',pair[1],',',pair[2],',',pair[3])
        
        if pair[1][0]==pair[2] or pair[1][1]==pair[2]:
            sum += 1
        
    print(sum / len(ar))

def f2(imgsrc):
    proc = gproc()

    cod = proc.process(imgsrc).reduce_noise().split_words().codate_aparitions()
    
    
    print(*proc.get_action_list())
    print()
    for w in proc.get_words(): print(w)
    print()
    for w in cod: print(w)
    print()
    
    words = np.array(proc.codate_aparitions())

    kmean = KMeans(n_clusters=2, random_state=0, n_init="auto")
    m = kmean.fit(words)
    print(m.labels_)
    
    words2 = np.array(proc.codate_duration())
    
    kmean2 = KMeans(n_clusters=2, random_state=0, n_init="auto")
    m2 = kmean2.fit(words2)
    print(m2.labels_)

def f3(imgsrc):
    proc = gproc()

    cod = proc.process(imgsrc)
    print(proc.get_action_list())
    print()
    
 
def plot_gaze_errors():
        parent_dir = os.path.dirname(r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\trying\gaze_estimations.csv')
        with open(r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\trying\gaze_estimations.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = {rows[0]: rows[1:] for rows in csv_reader}
        pitches = np.array(list(map(float, data['Original Pitch'])))
        yaws = np.array(list(map(float, data['Original Yaw'])))
        system_pitches = np.array(list(map(float, data['System Pitch'])))
        system_yaws = np.array(list(map(float, data['System Yaw'])))
        frames = list(range(len(pitches)))
        
        with open(r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\trying\pitch_and_yaw.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = {rows[0]: rows[1:] for rows in csv_reader}
        pf_pitches = np.array(list(map(float, data['PF Pitch'])))
        pf_yaws = np.array(list(map(float, data['PF Yaw'])))
        
        with open(r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\trying\gt.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = {rows[0]: rows[1:] for rows in csv_reader}
        gt_pitches = np.array(list(map(float, data['Ground Truth Pitch'])))
        gt_yaws = np.array(list(map(float, data['Ground Truth Yaw'])))
        
        mse_pitch_original = mean_squared_error(gt_pitches, pitches)
        mse_yaw_original = mean_squared_error(gt_yaws, yaws)
        
        mse_pitch_system = mean_squared_error(gt_pitches, system_pitches)
        mse_yaw_system = mean_squared_error(gt_yaws, system_yaws)
        
        mse_pitch_pf = mean_squared_error(gt_pitches, pf_pitches)
        mse_yaw_pf = mean_squared_error(gt_yaws, pf_yaws)
        
        # Calculate absolute error for each frame
        asolute_errors_pitch_original = abs(gt_pitches - pitches)
        asolute_errors_yaw_original = abs(gt_yaws - yaws)
    
        asolute_errors_pitch_system = abs(gt_pitches - system_pitches)
        asolute_errors_yaw_system = abs(gt_yaws - system_yaws)
        
        asolute_errors_pitch_pf = abs(gt_pitches - pf_pitches)
        asolute_errors_yaw_pf = abs(gt_yaws - pf_yaws)

        # Plot MSE for Pitch Original
        plt.subplot(1, 3, 1)
        plt.bar(frames, asolute_errors_pitch_original, color='blue', label='Original')
        plt.xlabel('Frame')
        plt.ylabel('Absolute error')
        plt.legend()
        plt.grid(axis='y')
        # Plot MSE for Pitch System
        plt.subplot(1, 3, 2)
        plt.bar(frames, asolute_errors_pitch_system, color='green', alpha=0.6, label='System')
        plt.xlabel('Frame')
        plt.ylabel('Absolute error')
        plt.legend()
        plt.grid(axis='y')
        # Plot MSE for Pitch PF
        plt.subplot(1, 3, 3)
        plt.bar(frames, asolute_errors_pitch_pf, color='red', alpha=0.6, label='PF')
        plt.xlabel('Frame')
        plt.ylabel('Absolute error')
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show() 

def main():
    
    # imgsrc = DrivfaceInput((1,2,3))
    
    # f4(imgsrc)
    # print("3A 4V")
    imgsrc = DrivfaceInput((1,2,3,4))
    proc = gproc()
    #proc.render(imgsrc)
    proc.render(imgsrc, savePath=r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\trying\image_{imNr}.png')
    
    #f4(imgsrc)

    # drvalidation()

    return 0

if __name__ == "__main__":
    main()
