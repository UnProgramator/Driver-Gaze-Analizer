# loading library 
import cv2 
import numpy as np 

base_path = 'D:/Programming/Datasets/drivface/DrivFace/DrivImages/DrivImages'

img = cv2.imread(base_path+'/20130529_01_Driv_013_f .jpg') 

cv2.imshow('car_orig.jpg', img)

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
    vertical_mb = cv2.filter2D(img, -1, kernel_v) 

    # Apply the horizontal kernel. 
    #horizonal_mb = cv2.filter2D(img, -1, kernel_h) 

    # Save the outputs. 
    cv2.imwrite('D:/car_vertical{}.jpg'.format(i), vertical_mb) 
    # cv2.imwrite('car_horizontal.jpg', horizonal_mb)
    cv2.waitKey(200)

    return vertical_mb

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

def my_bluri():
    
    img_b = bluri(20)

    mask = np.zeros(img_b.shape)

    _x,_y = img_b.shape

    _x //= 3

    mask = cv2.rectangle(mask, (_x, 0), (2*_x, _y), (255,), -1)

    out = np.where(mask==(255, 255, 255), img, img_b)

    cv2.imshow('', out)

my_bluri()



#for i in range(5, 30, 5):
#    bluri(i)
