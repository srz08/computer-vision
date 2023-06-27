import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

def convolve(I, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(I)
    #pad image based on kernel size
    I_padded = np.zeros((I.shape[0] + 2 * (kernel.shape[0] // 2), I.shape[1] + 2 * (kernel.shape[1] // 2)))
    I_padded[(kernel.shape[0] // 2):-(kernel.shape[0] // 2), (kernel.shape[1] // 2):-(kernel.shape[1] // 2)] = I
    for x in range(I.shape[1]):
        for y in range(I.shape[0]):
            output[y, x] = (kernel * I_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
    return output

def gaussian_smoothing(I, N, sigma):
    kernel = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            kernel[i, j] = np.exp(-((i - N // 2) ** 2 + (j - N // 2) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return convolve(I, kernel)

def image_gradient(I):
    I = np.float64(I)
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    I_x = convolve(I, kernel_x)
    I_y = convolve(I, kernel_y)
    
    magnitude = np.sqrt(I_x ** 2 + I_y ** 2)
    magnitude *= 255.0 / magnitude.max()
    angle = np.arctan2(I_y, I_x)
    return magnitude, angle



def gradient_magnitude_histogram(magnitude, percentage_of_non_edge):
    histogram = np.zeros(256)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            histogram[int(magnitude[i, j])] += 1
    histogram /= np.sum(histogram)
    # plt.plot(histogram)
    # plt.show()
    for i in range(1, 256):
        histogram[i] += histogram[i - 1]
    threshold = 0
    for i in range(255, -1, -1):
        if histogram[i] < percentage_of_non_edge:
            threshold = i
            break
    return threshold

def non_maximum_suppression(magnitude, angle):
    #use interpolation with I(p) = alpha*I(p1) + (1-alpha)*I(p2) and return the magnitude
    new_mag = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if angle[i, j] < 0:
                angle[i, j] += np.pi
            if 0 <= angle[i, j] < np.pi / 8 or 7 * np.pi / 8 <= angle[i, j] <= np.pi:
                alpha = np.tan(angle[i, j])
                p1 = magnitude[i, j + 1]
                p2 = magnitude[i, j - 1]
            elif np.pi / 8 <= angle[i, j] < 3 * np.pi / 8:
                alpha = 1 / np.tan(angle[i, j])
                p1 = magnitude[i + 1, j - 1]
                p2 = magnitude[i - 1, j + 1]
            elif 3 * np.pi / 8 <= angle[i, j] < 5 * np.pi / 8:
                alpha = np.tan(angle[i, j])
                p1 = magnitude[i + 1, j]
                p2 = magnitude[i - 1, j]
            else:
                alpha = 1 / np.tan(angle[i, j])
                p1 = magnitude[i + 1, j + 1]
                p2 = magnitude[i - 1, j - 1]
            new_mag[i, j] = magnitude[i, j] if magnitude[i, j] >= alpha * p1 + (1 - alpha) * p2 else 0
    return new_mag


def thresholding(magnitude, low_threshold, high_threshold):
    #return the binary image
    threshold_mag = np.zeros_like(magnitude)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if magnitude[i, j] >= high_threshold:
                threshold_mag[i, j] = 1
            elif magnitude[i, j] >= low_threshold:
                threshold_mag[i, j] = 2
    return threshold_mag

def recursive_linking(image, magnitude, i, j, visited):
    if i < 0 or i >= magnitude.shape[0] or j < 0 or j >= magnitude.shape[1] or (i, j) in visited:
        return
    visited[(i, j)] = True
    if magnitude[i, j] == 1:
        image[i, j] = 1
        print("hey")
        recursive_linking(image, magnitude, i + 1, j, visited)
        recursive_linking(image, magnitude, i - 1, j, visited)
        recursive_linking(image, magnitude, i, j + 1, visited)
        recursive_linking(image, magnitude, i, j - 1, visited)
        recursive_linking(image, magnitude, i + 1, j + 1, visited)
        recursive_linking(image, magnitude, i + 1, j - 1, visited)
        recursive_linking(image, magnitude, i - 1, j + 1, visited)
        recursive_linking(image, magnitude, i - 1, j - 1, visited)

def run_connect(image, i, j):
    if i < 0 or i >= image.shape[0] or j < 0 or j >= image.shape[1]:
        return False
    if image[i, j] == 1:
        return True
    image[i, j] = 1
    return run_connect(image, i + 1, j) or run_connect(image, i - 1, j) or run_connect(image, i, j + 1) or run_connect(image, i, j - 1) or run_connect(image, i + 1, j + 1) or run_connect(image, i + 1, j - 1) or run_connect(image, i - 1, j + 1) or run_connect(image, i - 1, j - 1)


def edge_linking(magnitude):
    image = np.zeros_like(magnitude)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if magnitude[i, j] == 1:
                visited = {}
                recursive_linking(image, magnitude, i, j, visited)


    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if magnitude[i, j] == 2:
                connect_strong = run_connect(image, i, j)
                if connect_strong:
                    image[i, j] = 1
    return image

# def sobel_filters(img):
#     Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
#     Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
#     Ix = ndimage.filters.convolve(img, Kx)
#     Iy = ndimage.filters.convolve(img, Ky)
    
#     G = np.hypot(Ix, Iy)
#     G = G / G.max() * 255
#     theta = np.arctan2(Iy, Ix)
    
#     return G, theta

# def non_max_suppression(img, D):
#     M, N = img.shape
#     Z = np.zeros((M,N), dtype=np.int32)
#     angle = D * 180. / np.pi
#     angle[angle < 0] += 180

    
#     for i in range(1,M-1):
#         for j in range(1,N-1):
#             try:
#                 q = 255
#                 r = 255
                
#                #angle 0
#                 if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
#                     q = img[i, j+1]
#                     r = img[i, j-1]
#                 #angle 45
#                 elif (22.5 <= angle[i,j] < 67.5):
#                     q = img[i+1, j-1]
#                     r = img[i-1, j+1]
#                 #angle 90
#                 elif (67.5 <= angle[i,j] < 112.5):
#                     q = img[i+1, j]
#                     r = img[i-1, j]
#                 #angle 135
#                 elif (112.5 <= angle[i,j] < 157.5):
#                     q = img[i-1, j-1]
#                     r = img[i+1, j+1]

#                 if (img[i,j] >= q) and (img[i,j] >= r):
#                     Z[i,j] = img[i,j]
#                 else:
#                     Z[i,j] = 0

#             except IndexError as e:
#                 pass
    
#     return Z

# def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
#     highThreshold = img.max() * highThresholdRatio;
#     lowThreshold = highThreshold * lowThresholdRatio;
    
#     M, N = img.shape
#     res = np.zeros((M,N), dtype=np.int32)
    
#     weak = np.int32(25)
#     strong = np.int32(255)
    
#     strong_i, strong_j = np.where(img >= highThreshold)
#     zeros_i, zeros_j = np.where(img < lowThreshold)
    
#     weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
#     res[strong_i, strong_j] = strong
#     res[weak_i, weak_j] = weak
    
#     return (res, weak, strong)


# def hysteresis(img, weak, strong=255): 
#         M, N = img.shape  
#         for i in range(1, M-1):
#             for j in range(1, N-1):
#                 if (img[i,j] == weak):
#                     try:
#                         if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
#                             or (img[i, j-1] == strong) or (img[i, j+1] == strong)
#                             or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
#                             img[i, j] = strong
#                         else:
#                             img[i, j] = 0
#                     except IndexError as e:
#                         pass
#         return img

# def gaussian_kernel(size, sigma=1):
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * np.pi * sigma**2)
#     g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#     return g

# image = cv2.imread('lena.bmp', 0)
# image = image.astype(np.float32)
# # image = image / 255.0
# #use gaussian kernel to smooth the image 
# g = gaussian_kernel(5, 1)
# image = cv2.filter2D(image, -1, g)
# #compute the gradient magnitude and angle using sobel_filters function
# magnitude, angle = sobel_filters(image)
# #apply non-maximum suppression to the magnitude image
# magnitude = non_max_suppression(magnitude, angle)
# #apply double thresholding to the magnitude image
# threshold_mag = thresholding(magnitude, 0.01, 0.001)
# #apply edge linking to the threshold image using hysteresis
# edges = hysteresis(threshold_mag, 25, 255)

# #plot edges over the original image
# # image = image*edges
# plt.imshow(edges, cmap='gray')
# plt.show()






image = cv2.imread('lena.bmp', 0)

image = gaussian_smoothing(image, 5, 1)
Magnitude, Angle = image_gradient(image)
high_threshold = gradient_magnitude_histogram(Magnitude, 0.91)
low_threshold = 0.5 * high_threshold
Magnitude = non_maximum_suppression(Magnitude, Angle)
threshold_mag = thresholding(Magnitude, low_threshold, high_threshold)
edges = edge_linking(threshold_mag)
#write image to txt
np.savetxt('image.txt', image, fmt='%d')


#plot edges over the original image
# image = image*edges
plt.imshow(edges, cmap='gray')
plt.show()





