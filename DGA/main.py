from calendar import c
from math import sqrt
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
        # parent_dir = os.path.dirname(r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\results\trying\gaze_estimations.csv')
        parent_dir = os.path.dirname(r'results\trying\gaze_estimations.csv')
        with open(r'results\trying\gaze_estimations.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = {rows[0]: rows[1:] for rows in csv_reader}
        pitches_nn = np.array(list(map(float, data['Original Pitch'])))
        yaws_nn = np.array(list(map(float, data['Original Yaw'])))
        system_pitches = np.array(list(map(float, data['System Pitch'])))
        system_yaws = np.array(list(map(float, data['System Yaw'])))
        frames = list(range(len(pitches_nn)))
        
        with open(r'results\trying\pitch_and_yaw.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = {rows[0]: rows[1:] for rows in csv_reader}
        pf_pitches = np.array(list(map(float, data['PF Pitch'])))
        pf_yaws = np.array(list(map(float, data['PF Yaw'])))
        
        with open(r'results\trying\gt.csv', 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = {rows[0]: rows[1:] for rows in csv_reader}
        gt_pitches = np.array(list(map(float, data['Ground Truth Pitch'])))
        gt_yaws = np.array(list(map(float, data['Ground Truth Yaw'])))
        
        mse_pitch_original = mean_squared_error(gt_pitches, pitches_nn)
        mse_yaw_original = mean_squared_error(gt_yaws, yaws_nn)
        
        mse_pitch_system = mean_squared_error(gt_pitches, system_pitches)
        mse_yaw_system = mean_squared_error(gt_yaws, system_yaws)
        
        # mse_pitch_pf = mean_squared_error(gt_pitches, pf_pitches)
        # mse_yaw_pf = mean_squared_error(gt_yaws, pf_yaws)

        print(mse_pitch_original, mse_yaw_original)
        print(mse_pitch_system, mse_yaw_system)
        # print(mse_pitch_pf, mse_yaw_pf)
        
        # Calculate absolute error for each frame
        asolute_errors_pitch_nn = abs(gt_pitches - pitches_nn)
        asolute_errors_yaw_nn = abs(gt_yaws - yaws_nn)
    
        asolute_errors_pitch_system = abs(gt_pitches - system_pitches)
        asolute_errors_yaw_system = abs(gt_yaws - system_yaws)
        
        # asolute_errors_pitch_pf = abs(gt_pitches - pf_pitches)
        # asolute_errors_yaw_pf = abs(gt_yaws - pf_yaws)

        # total_error_pitch_nn = np.power(gt_pitches - pitches_nn,2)
        # total_error_yaw_nn = np.power(gt_yaws - yaws_nn,2)
        # total_error_nn = np.sqrt(total_error_pitch_nn + total_error_yaw_nn)

        # total_error_pitch_sys = np.power(gt_pitches - system_pitches,2)
        # total_error_yaw_sys = np.power(gt_yaws - system_yaws,2)
        # total_error_sys = np.sqrt(total_error_pitch_sys + total_error_yaw_sys)
        
        # total_error_pitch_pf = np.power(gt_pitches - pf_pitches,2)
        # total_error_yaw_pf = np.power(gt_yaws - pf_yaws,2)
        # total_error_pf = np.sqrt(total_error_pitch_pf + total_error_yaw_pf)
        


        val_ = 1.4 # the height of the graph on 0Y

        c = 2
        l = 2

        # Plot MSE for Pitch Original
        plt.subplot(c, l, 1)
        plt.bar(frames, asolute_errors_pitch_nn, color='blue', label='Original')
        plt.xlabel('Frame')
        plt.ylabel('Absolute error')
        plt.legend()
        plt.grid(axis='y')
        plt.ylim(top=val_)
        # Plot MSE for Pitch System
        plt.subplot(c, l, 2)
        plt.bar(frames, asolute_errors_pitch_system, color='green',  label='System')
        plt.xlabel('Frame')
        plt.ylabel('Absolute error')
        plt.legend()
        plt.grid(axis='y')
        plt.ylim(top=val_)
        # Plot MSE for Pitch PF
        # plt.subplot(c, l, 3)
        # plt.bar(frames, asolute_errors_pitch_pf, color='red',  label='PF')
        # plt.xlabel('Frame')
        # plt.ylabel('Absolute error')
        # plt.legend()
        # plt.grid(axis='y')
        # plt.ylim(top=val_)

        # Plot MSE for Pitch Original
        plt.subplot(c, l, 3)
        plt.bar(frames, asolute_errors_yaw_nn, color='blue', label='Original')
        plt.xlabel('Frame')
        plt.ylabel('Absolute error')
        plt.legend()
        plt.grid(axis='y')
        plt.ylim(top=val_)
        # Plot MSE for Pitch System
        plt.subplot(c, l, 4)
        plt.bar(frames, asolute_errors_yaw_system, color='green',  label='System')
        plt.xlabel('Frame')
        plt.ylabel('Absolute error')
        plt.legend()
        plt.grid(axis='y')
        plt.ylim(top=val_)
        # Plot MSE for Pitch PF
        # plt.subplot(c, l, 6)
        # plt.bar(frames, asolute_errors_yaw_pf, color='red',  label='PF')
        # plt.xlabel('Frame')
        # plt.ylabel('Absolute error')
        # plt.legend()
        # plt.grid(axis='y')
        # plt.ylim(top=val_)

        # # Plot MSE for Pitch Original
        # plt.subplot(c, l, 7)
        # plt.bar(frames, total_error_nn, color='blue', label='Original')
        # plt.xlabel('Frame')
        # plt.ylabel('Absolute error')
        # plt.legend()
        # plt.grid(axis='y')
        # plt.ylim(top=val_)
        # # Plot MSE for Pitch System
        # plt.subplot(c, l, 8)
        # plt.bar(frames, total_error_sys, color='green',  label='System')
        # plt.xlabel('Frame')
        # plt.ylabel('Absolute error')
        # plt.legend()
        # plt.grid(axis='y')
        # plt.ylim(top=val_)
        # # Plot MSE for Pitch PF
        # plt.subplot(c, l, 9)
        # plt.bar(frames, total_error_pf, color='red',  label='PF')
        # plt.xlabel('Frame')
        # plt.ylabel('Absolute error')
        # plt.legend()
        # plt.grid(axis='y')
        # plt.ylim(top=val_)


        


        #plt.tight_layout()
        plt.show() 

def main():
    
    # imgsrc = DrivfaceInput((1,2,3))
    
    # f4(imgsrc)
    # print("3A 4V")
    
    #proc.render(imgsrc)
    #proc.render(imgsrc, savePath=r'results\trying\image_{imNr}.png')
    
    #f4(imgsrc)

    # drvalidation()

    #plot_gaze_errors()
    
    # imgsrc = DrivfaceInput((1,2,3,4))
    # proc = gproc()
    # proc.saveResults(imgsrc, savePath=r'results\trying\gaze_estimations.csv')

    plot_gaze_errors()

    return 0

if __name__ == "__main__":
    main()
