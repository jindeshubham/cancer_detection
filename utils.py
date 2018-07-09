import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.interactive(False)
import numpy as np
import os


#preprocessing involves converting image to grayscale and applying median filter
def preprocessing(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.medianBlur(image,5)
    return image


#applying threshold to seperate background from image
def segmenting(image):
    retval, threshold= cv2.threshold(image,120,220,cv2.THRESH_BINARY)
    return threshold

def resizing(image):
    image = np.resize(image,(256,256))
    return image


def normalize (image):
    image = float(image/255)
    return image