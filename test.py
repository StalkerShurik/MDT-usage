import cv2 as cv
import numpy as np

from PIL import Image

import mahalanobis_transformation


img = cv.imread("DOG.jpg") #read image

ret, img = cv.threshold(img, 127, 255, 0) # pixels > 127 becomes 255

img = np.where(img == 255, 1, img) # all 255 becomes 1

trans = np.array([[1, 0], [0, 1]]) #matrix induces Euclidean metric (you can use any positive-definite matrix)

#params: image, transformation matrix, connectivity_type, is_signed
#connectivity_type: 8-connectivity and 4-connectivity for 2d images, 26-connectivity and 6-connectivity for 3d
transformed = mahalanobis_transformation.MDT_connectivity(img, trans, "8-connectivity", 0)

#brute algo: params same without connectivity
#transformed = mahalanobis_transformation.MDT_brute(img, trans, 0)

im = Image.fromarray(np.abs(transformed))
im.show()
