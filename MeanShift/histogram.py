import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

#https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
def histogram(image):
    img = cv2.imread(f"./Data/{image}.jpg", cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img_output_rgb = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

    return img_output_rgb
