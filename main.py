import cv2
import numpy as np

p1im1 = cv2.imread('./images/p1im1.png', cv2.IMREAD_GRAYSCALE)
p1im4 = cv2.imread('./images/p1im4.png', cv2.IMREAD_GRAYSCALE)
p1im5 = cv2.imread('./images/p1im5.png', cv2.IMREAD_GRAYSCALE)
p1im6 = cv2.imread('./images/p1im6.png', cv2.IMREAD_GRAYSCALE)

prewittX = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewittY = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

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

def getGaussian(size):
    gaussian = np.zeros([size, size])
    sigma = 1
    r = size // 2
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            denom = 2 * np.pi * (sigma ** 2)
            exp = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
            gaussian[i+r][j+r] =  exp / denom
    return gaussian

def getLoG(size, sigma):
    LoG = np.zeros([size, size])
    r = size // 2
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            num = (i ** 2 + j ** 2) - (2 * sigma ** 2)
            denom = 2 * np.pi * (sigma ** 6)
            exp = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
            LoG[i+r][j+r] = num * exp / denom
    return LoG

def getGradient(X, Y):
    height = X.shape[0]
    width = X.shape[1]
    gradient = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            gradient[i][j] = np.abs(X[i][j]) + np.abs(Y[i][j])
            if gradient[i][j] > 255:
                gradient[i][j] = 255
            elif gradient[i][j] < 0:
                gradient[i][j] = 0
    return gradient

def getGradientMD(X, Y):
    height = X.shape[0]
    width = X.shape[1]
    magnitude = np.zeros([height, width])
    degree = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            magnitude[i][j] = np.sqrt(np.square(X[i][j]) + np.square(Y[i][j]))
            degree[i][j] = np.arctan(Y[i][j] / X[i][j])
    return magnitude, degree
    
def nonMaximalSupress(image, degree):
    height = image.shape[0]
    width = image.shape[1]
    NMS = np.copy(image)
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                NMS[i][j] = 0
            else:
                direction = degree[i][j] % 4
                if direction == 0:
                    if image[i][j] <= image[i][j-1] or image[i][j] <= image[i][j+1]:
                            NMS[i][j] = 0
                if direction == 1:
                    if image[i][j] <= image[i-1][j+1] or image[i][j] <= image[i+1][j-1]:
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
    for i in range(height - 1):
        for j in range(width - 1):
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
    newImage = np.zeros([height, width])
    if row == col:
        border = int(row / 2)
        for i in range(border, height - border):
            for j in range(border, width - border):
                newImage[i][j] = np.sum(image[i-border : i+row-border, j-border : j+row-border] * filter)
    return newImage

def convolution2d_norm(image, filter):
    height = image.shape[0]
    width = image.shape[1]
    row, col = filter.shape
    newImage = np.zeros([height, width])
    if row == col:
        border = int(row / 2)
        for i in range(border, height - border):
            for j in range(border, width - border):
                newImage[i][j] = np.sum(image[i-border : i+row-border, j-border : j+row-border] * filter)
    max = np.max(newImage)
    min = np.min(newImage)
    if min < 0: 
        result = np.uint8((newImage + -min) * 255 / np.max(newImage + -min))
    else:
        result = np.uint8(newImage * 255 / np.max(newImage))
    return result

def applyThreshold(image, threshold):
    height = image.shape[0]
    width = image.shape[1]
    magnitude = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            if image[i][j] > threshold:
                magnitude[i][j] = 255
            else:
                magnitude[i][j] = 0
    return magnitude

def applyPrewitt(image, threshold):
    X = convolution2d_norm(image, prewittX)
    Y = convolution2d_norm(image, prewittY)
    gradient = getGradient(X, Y)
    return applyThreshold(gradient, threshold)

def applySobel(image, threshold):
    X = convolution2d_norm(image, sobelX)
    Y = convolution2d_norm(image, sobelY)
    gradient = getGradient(X, Y)
    return applyThreshold(gradient, threshold)

def applyLoG(image, size, sigma, threshold):
    LoG = getLoG(size, sigma)
    newImage = convolution2d_norm(image, LoG)
    return applyThreshold(newImage, threshold)

def applyLoGPre(image, size, sigma, threshold):
    gaussian = getGaussian(3)
    blurred = convolution2d_norm(image, gaussian)
    LoG = getLoG(size, sigma)
    newImage = convolution2d_norm(blurred, LoG)
    return applyThreshold(newImage, threshold)

def canny(image, lt, ht):
    gaussian = getGaussian(3)
    blurred = convolution2d(image, gaussian)
    X = convolution2d(blurred, sobelX)
    Y = convolution2d(blurred, sobelY)
    magnitude, degree = getGradientMD(X, Y)
    NMS = nonMaximalSupress(magnitude, degree)
    result = doubleThreshold(NMS, lt, ht)
    return result

p1im1_result = applyPrewitt(p1im1, 250)
cv2.imshow('1-1-1', p1im1_result)

# p1im4_result = applyPrewitt(p1im4, 230)
# cv2.imshow('4-1-1', p1im4_result)

# p1im5_result = applyPrewitt(p1im5, 250)
# cv2.imshow('5-1-1', p1im5_result)

# p1im6_result = applyPrewitt(p1im6, 230)
# cv2.imshow('6-1-1', p1im6_result)
# cv2.waitKey(0)


# p1im1_result = applySobel(p1im4, 230)
# cv2.imshow('1-1-2', p1im1_result)
# p1im1_result = applyLoG(p1im1, 5, 0.7, 0)
# cv2.imshow('1-1-3', p1im1_result)
# p1im1_result = applyLoGPre(p1im1, 5, 0.7, 0)
# cv2.imshow('1-1-4', p1im1_result)
# cv2.waitKey(0)

# p1im5_result = applyPrewitt(p1im5, 230)
# cv2.imshow('4-1', p1im5_result)

# p1im5_result = applySobel(p1im5, 230)
# cv2.imshow('4-2', p1im5_result)
# cv2.waitKey(0)



# p1im4_result = applySobel(p1im4, 230)
# cv2.imshow('4-1-2', p1im4_result)

# p1im4_result = applyLoG(p1im4, 5, 0.7, 130)
# cv2.imshow('4-1-3', p1im4_result)

# p1im4_result = applyLoGPre(p1im4, 5, 0.7, 164)
# cv2.imshow('4-1-4', p1im4_result)

# p1im4_result = canny(p1im4, -20, 10)
# cv2.imshow('4-2', p1im4_result)
# cv2.waitKey(0)

# p1im5_result = applyLoG(p1im5, 5, 0.7, 160)
# cv2.imshow('5-1', p1im5_result)
# cv2.waitKey(0)

# p1im5_result = applyLoGPre(p1im5, 5, 0.7, 200)
# cv2.imshow('5-2', p1im5_result)
# cv2.waitKey(0)

# p1im5_result = canny(p1im5, 0, 10)
# cv2.imshow('5-3', p1im5_result)
# cv2.waitKey(0)