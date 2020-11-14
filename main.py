import cv2
import numpy as np

p1im1 = cv2.imread('./images/p1im1.png', cv2.IMREAD_COLOR)
p1im2 = cv2.imread('./images/p1im2.png', cv2.IMREAD_COLOR)
p1im3 = cv2.imread('./images/p1im3.png', cv2.IMREAD_COLOR)
p1im4 = cv2.imread('./images/p1im4.png', cv2.IMREAD_COLOR)
p1im5 = cv2.imread('./images/p1im5.png', cv2.IMREAD_COLOR)
p1im6 = cv2.imread('./images/p1im6.png', cv2.IMREAD_GRAYSCALE)

sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobelY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

def getLoG(size, sigma):
    LOG = np.zeros([size, size])
    r = size // 2
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            d1 = 2 * np.pi * (sigma ** 2)
            d2 = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
            d3 = (i ** 2 + j ** 2 - (2 * sigma ** 2)) / (sigma ** 4)
            LOG[i+r][j+r] = d2 * d3 / d1
    return LOG

def getGaussian(size):
    gaussian = np.zeros([size, size])
    sigma = 1
    r = size // 2
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            d1 = 2 * np.pi * (sigma**2)
            d2 = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
            gaussian[i+r, j+r] =  d2 / d1
    return gaussian

def getGradient(X, Y):
    height = X.shape[0]
    width = X.shape[1]
    magnitude = np.zeros([height, width])
    degree = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            magnitude[i][j] = np.sqrt(np.square(X[i, j]) + np.square(Y[i, j]))
            degree[i][j] = np.arctan(Y[i][j] / X[i][j])
    return magnitude, degree
    
def nonMaximalSupress(image, degree):
    height = image.shape[0]
    width = image.shape[1]
    NMS = np.copy(image)
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height -1 or j == 0 or j == width -1:
                NMS[i][j] = 0
                continue
            direction = degree[i][j] % 4
            if direction == 0:
               if image[i][j] <= image[i][j-1] or image[i][j] <= image[j][i+1]:
                    NMS[i][j] = 0
            if direction == 1:
               if image[i][j] <= image[i-1][j+1] or image[i][j] <= image[j+1][i-1]:
                    NMS[i][j] = 0
            if direction == 2:
                 if image[i][j] <= image[i-1][j] or image[i][j] <= image[i+1][j]:
                    NMS[i][j] = 0
            if direction == 3:
                 if image[i][j] <= image[i-1][j-1] or image[i][j] <= image[i+1][j+1]:
                    NMS[i][j] = 0
    return NMS

def doubleThreshold(NMS, lt, ht):
    height = NMS.shape[0]
    width = NMS.shape[1]
    t = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            if NMS[i][j] > ht:
                t[i][j] = 255
            elif NMS[i][j] >= lt and NMS[i][j] <= ht:
                t[i][j] = 75
            elif NMS[i][j] < lt:
                t[i][j] = 0
    result = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            if t[i][j] == 75:
                if t[i+1][j] == 255 or t[i-1][j] == 255 or t[i][j+1] == 255 or t[i][j-1] == 255 or t[i+1][j+1] == 255 or t[i-1][j-1] == 255:
                    result[i][j] = 255
                else:
                    result[i][j] = 0
    return result

def convolution2d(image, filter):
    height = image.shape[0]
    width = image.shape[1]
    row, col = filter.shape
    newImage = np.zeros(image.shape)
    if (row == col):
        border = int(row / 2)
        for i in range(border, height - border):
            for j in range(border, width - border):
                newImage[i][j] = np.sum(image[i-border : i+row-border, j-border : j+row-border] * filter)
    return newImage

def applyLOG(image, size, sigma):
    LOG = getLoG(size, sigma)
    print(LOG)
    return convolution2d(image, LOG)

def canny(image, lt, ht):
    gaussian = getGaussian(3)
    blurred = convolution2d(image, gaussian)
    X = convolution2d(blurred, sobelX)
    Y = convolution2d(blurred, sobelY)
    magnitude, degree = getGradient(X, Y) #, direction = getGradient(X, Y)
    NMS = nonMaximalSupress(magnitude, degree)
    result = doubleThreshold(NMS, lt, ht)
    return result

# p1 = canny(p1im6, 20, 80)
# cv2.imshow('1', p1)
# cv2.waitKey(0)

p1 = applyLOG(p1im6, 5, 1)
cv2.imshow('1', p1)
cv2.waitKey(0)

# p1im1_result = 
# cv2.imshow('1', p1im1_result)
# cv2.waitKey(0)

# p1im2_result = 
# cv2.imshow('2', p1im2_result)
# cv2.waitKey(0)

# p1im3_result = 
# p1im3_result = gammaCorrection(p1im3_result, 2.4)
# cv2.imshow('3', p1im3_result)
# cv2.waitKey(0)

# p1im4_result = 
# cv2.imshow('4', p1im4_result)
# cv2.waitKey(0)

# p1im5_result = 
# cv2.imshow('5', p1im5_result)
# cv2.waitKey(0)

# p1im5_result = 
# cv2.imshow('5', p1im5_result)
# cv2.waitKey(0)

# p1im6_result = 
# cv2.imshow('6', p1im6_result)
# cv2.waitKey(0)
