from backbone.processor import Processor
from .DreyeveValidation import DreyeveValidation
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

def drvalidation():
    dv = DreyeveValidation()
    
    points = dv.load_files([r'C:\Users\dpatrut\source\repos\Driver-Gaze-Analizer\DGA\dataset\dreyeve\etg_samples_{}.txt'.format(i) for i in range(1,5)])
    
    # no up/down verification
    proc = Processor(480,1440,0,1080,1,30) #30 frames = 1 sec, video la 30 fps 1080p: 1920x1080, 1920:4=480 480l|960f|480r l:[0-480], f:(480,1440), r:[1440,1920] 
    
    #with up/down verification: screen divided into 5 sections, the first is for up the last is for down 1080/5=216 up-[0, 216] up-[864,1080] 
    #proc = Processor(480,1440,216,864,1,30) #30 frames = 1 sec, video la 30 fps 1080p: 1920x1080, 1920:4=480 480l|960f|480r l:[0-480], f:(480,1440), r:[1440,1920] 
    
    #with up/down verification: screen divided into 4 sections, the first is for up the last is for down 1080/5=270 up-[0, 270] up-[810,1080] 
    #proc = Processor(480,1440,270,810,1,30) #30 frames = 1 sec, video la 30 fps 1080p: 1920x1080, 1920:4=480 480l|960f|480r l:[0-480], f:(480,1440), r:[1440,1920] 
    
    cod = proc.load_array(points).reduce_noise().split_words().codate_duration()

    print(*proc.get_action_list())
    print()
    for w in proc.get_words(): print(w)
    print()
    for w in cod: print(w)
    print(len(cod))
    
    w_array = np.array(cod)
    
    siluet(w_array)
    

def siluet(w_array):
    fig, ax = plt.subplots(3,3, figsize=(20,12))
    vals = [3,4,5,6,7,8,9,10]
    for i in vals:
        '''
        Create KMeans instances for different number of clusters
        '''
        km = KMeans(n_clusters=i, random_state=0, n_init="auto")
        q, mod = divmod(i-vals[0], 3)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[mod][q])
        ax[mod][q].set_title('K={}'.format(i))
        visualizer.fit(w_array) 
      
        
    fig.show()
    fig.waitforbuttonpress(1000)    

def elbow(w_array):
    km = KMeans(n_clusters=7, random_state=0, n_init="auto")
    
    #km.fit(w_array)
    
    visualizer = KElbowVisualizer(km, k=(2,12))
 
    visualizer.fit(w_array)        # Fit the data to the visualizer
    visualizer.show()
