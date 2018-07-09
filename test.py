# import cv2
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.interactive(False)
# import numpy as np
# import os
#
# import pywt
# #loads the image
# image = cv2.imread('./data/train/melanoma/ISIC_0000002.jpg')
#
# #cv2.imshow("color",image)
# #convert ot grayscale
# image = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
#
# # plt.show()
#
# #median blur
#
# image = cv2.medianBlur(image,5)
# # plt.imshow(image,cmap='gray')
# # plt.show()
#
# #Thresholding
# retval, threshold = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
#
# image = np.resize(threshold,(250,250))
#
# print(image.shape)
#
# image = np.reshape(image,(-1))
#
# print (image.shape)

# cv2.imshow("image",image)
# cv2.imshow("threshold",threshold)


# coeffs = pywt.dwt2(threshold, 'haar')
#
# print (len(coeffs))
#
# for coeff in coeffs:
#     print ("Coefficent",coeff)
#
# folder = './data/train/melanoma/'
# c=0
# for filename in os.listdir(folder):
#     print(os.path.join(folder,filename))
#     img = cv2.imread(os.path.join(folder,filename))
#
#     if img is not None:
#         c +=1
# print (c)


Y = ['melanoma','nevus','nevus','seborrheic_keratosis']
mapping = {"melanoma": 1, "nevus": 2, "seborrheic_keratosis": 3}
Y = [mapping[y] for y in Y]
print (Y)