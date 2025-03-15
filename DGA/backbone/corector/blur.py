# loading library 
from pathlib import Path
import cv2 
from cv2.typing import MatLike, Range
import numpy as np 
import os, random

def main_test():
    base_path = 'D:/Programming/Datasets/drivface/DrivFace/DrivImages/DrivImages'

    img1 = cv2.imread(base_path+'/20130529_01_Driv_013_f .jpg') 

    cv2.imshow('car_orig.jpg', img1)

    print('itermediar')

    def bluri(kernel_size = 30) -> cv2.MatLike:

        # Specify the kernel size. 
        # The greater the size, the more the motion. 
        #kernel_size = 30

        # Create the vertical kernel. 
        kernel_v = np.zeros((kernel_size, kernel_size)) 

        # Create a copy of the same for creating the horizontal kernel. 
        # kernel_h = np.copy(kernel_v) 

        # Fill the middle row with ones. 
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        # kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 

        # Normalize. 
        kernel_v /= kernel_size 
        #kernel_h /= kernel_size 

        # Apply the vertical kernel. 
        vertical_mb = cv2.filter2D(img1, -1, kernel_v) 

        # Apply the horizontal kernel. 
        #horizonal_mb = cv2.filter2D(img, -1, kernel_h) 

        # Save the outputs. 
        cv2.imwrite('D:/car_vertical{}.jpg'.format(i), vertical_mb) 
        # cv2.imwrite('car_horizontal.jpg', horizonal_mb)
        cv2.waitKey(200)

        return vertical_mb

    def my_bluri():
    
        img_b = bluri(20)

        mask = np.zeros(img_b.shape)

        _x,_y = img_b.shape

        _x //= 3

        mask = cv2.rectangle(mask, (_x, 0), (2*_x, _y), (255,), -1)

        out = np.where(mask==(255, 255, 255), img1, img_b)

        cv2.imshow('', out)

    my_bluri()

    #for i in range(5, 30, 5):
    #    bluri(i)


@DeprecationWarning
def get_motion_blur_kernel(x, y, thickness=1, ksize=21):
    """ Obtains Motion Blur Kernel
        Inputs:
            x - horizontal direction of blur
            y - vertical direction of blur
            thickness - thickness of blur kernel line
            ksize - size of blur kernel
        Outputs:
            blur_kernel
        """
    blur_kernel = np.zeros((ksize, ksize))
    c = int(ksize/2)

    blur_kernel = np.zeros((ksize, ksize))
    blur_kernel = cv2.line(blur_kernel, (c+x,c+y), (c,c), (255,), thickness)
    return blur_kernel

def blur_entire_pic(img:MatLike, kernel_size:int=15) -> MatLike:
    '''
        args:
            img:MatLike the picture to blur
            kernel_size:int -> The greater the size, the more the motion it generates
        returns:
            blured_img:MatLike the blured pic
    '''
    # Create the vertical kernel. 
    kernel_v = np.zeros((kernel_size, kernel_size)) 

    # Fill the middle row with ones. 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 

    # Normalize. 
    kernel_v /= kernel_size 

    # Apply the vertical kernel. 
    return cv2.filter2D(img, -1, kernel_v) 

def blur_batch(dest:Path, src:Path, imTemp:str, idx:Range, interval:int, shake:int = 0):

    rnd = random.SystemRandom()

    i1 = interval-shake
    i2 = interval+shake
    seq = -1
    last = 0

    print('begin----\n')

    with open(dest+'rep.log','a') as f:
        print(f'blur {imTemp}, interval {interval} with random offset of {shake}', file=f)

        for i in idx:
            imn = imTemp.format(i)
            print(imn)
            im = cv2.imread(src+imn)
            if seq == -1 and i-last>i1:
                if i-last==i2 or rnd.getrandbits(1):
                    last = i
                    seq = 2
                    print(f'blur {i}->{i+2}', file=f)
            if seq != -1:
                if seq == 1:
                    im = blur_entire_pic(im, 20)
                else:
                    im = blur_entire_pic(im, 10)
                seq -= 1
            cv2.imwrite(dest+imn, im)
            del im

    print('end----\n')

def main_drivface():
    src = 'D:/Programming/Datasets/drivface/DrivFace/DrivImages/DrivImages/'
    dest ='D:/Programming/Datasets/drivface/DrivFace_blur/'
    sets = {'01 {:03d}.jpg':range(1,180), '02 {:03d}.jpg':range(1,171), '03 {:03d}.jpg':range(1,168), '04 {:03d}.jpg':range(1,91)}

    for k,v in sets.items():
        blur_batch(dest, src, k, v, 20, 2)

if __name__ == '__main__':
    main_drivface()
