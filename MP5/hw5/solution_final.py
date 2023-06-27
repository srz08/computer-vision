import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def gaussian_smoothing(I, N, sigma):
    kernel = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            kernel[i, j] = np.exp(-((i - N // 2) ** 2 + (j - N // 2) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return convolve(I, kernel)

def convolve(I, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(I)
    I_padded = np.zeros((I.shape[0] + 2 * (kernel.shape[0] // 2), I.shape[1] + 2 * (kernel.shape[1] // 2)))
    I_padded[(kernel.shape[0] // 2):-(kernel.shape[0] // 2), (kernel.shape[1] // 2):-(kernel.shape[1] // 2)] = I
    for x in range(I.shape[1]):
        for y in range(I.shape[0]):
            output[y, x] = (kernel * I_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
    return output

def image_gradient(I, kernel_x, kernel_y):
    I = np.float64(I)
    I_x = convolve(I, kernel_x)
    I_y = convolve(I, kernel_y)
    
    magnitude = np.sqrt(I_x ** 2 + I_y ** 2)
    magnitude *= 255.0 / magnitude.max()
    angle = np.rad2deg(np.arctan2(I_y, I_x))
    return magnitude, angle
    

def non_maximum_suppression(magnitude, angle): 
    nms = np.zeros(magnitude.shape)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                th = max(magnitude[i, j - 1], magnitude[i, j + 1])
            elif (22.5 <= angle[i, j] < 67.5):
                th = max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
            elif (67.5 <= angle[i, j] < 112.5):
                th = max(magnitude[i - 1, j], magnitude[i + 1, j])
            else:
                th = max(magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
            if magnitude[i, j] >= th:
                nms[i, j] = magnitude[i, j]
    nms_max = nms.max()
    k = 255.0 / nms_max
    return nms * k

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

def thresholding(magnitude, low_threshold, high_threshold):
    threshold_mag = np.zeros_like(magnitude)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if magnitude[i, j] >= high_threshold:
                threshold_mag[i, j] = 255
            elif magnitude[i, j] >= low_threshold:
                threshold_mag[i, j] = 128
    return threshold_mag

def edge_linking(after_threshold):
    after_linking = np.zeros(after_threshold.shape)
    for i in range(after_threshold.shape[0]):
        for j in range(after_threshold.shape[1]):
            if after_threshold[i,j] == 255:
                after_linking[i,j] = 255
            if after_threshold[i,j] == 128:
                if after_threshold[i-1,j] == 255 or after_threshold[i+1,j] == 255 or after_threshold[i-1,j-1] == 255 or after_threshold[i+1,j-1] == 255 or after_threshold[i-1,j+1] == 255 or after_threshold[i+1,j+1] == 255 or after_threshold[i,j-1] == 255 or after_threshold[i,j+1] == 255:
                    after_linking[i,j] = 255
    return after_linking

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (16,4))

N = [3,5,7]
sigma = [1,2,3]
percentage_of_non_edge = [0.95]
kernels_x = {"sobel": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), "roberts": np.array([[1,0],[0,-1]])}
kernels_y = {"sobel": np.array([[1,2,1],[0,0,0],[-1,-2,-1]]), "roberts": np.array([[0,1],[-1,0]])}
kernel_names = ["sobel", "roberts"]

for p in percentage_of_non_edge:
    for n in N:
        for s in sigma:
            for kernel_name in kernel_names:
                fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (16,4))
                img_name = 'test1'
                img = cv2.imread('{}.bmp'.format(img_name))[:,:,1]
                #gaussian smoothing
                gaussian = gaussian_smoothing(img, n, s)
                #image gradient
                filter_x, filter_y = kernels_x[kernel_name], kernels_y[kernel_name]
                magnitude, angle = image_gradient(gaussian, filter_x, filter_y)
                #non maximum suppression
                nms = non_maximum_suppression(magnitude, angle)
                #find threshold
                high_threshold = gradient_magnitude_histogram(nms,p)
                low_threshold = high_threshold * 0.5
                #apply thresholding
                after_threshold = thresholding(nms, low_threshold, high_threshold)
                #edge nms
                output = edge_linking(after_threshold)
                #display images on each subplot in grayscale
                ax1.imshow(gaussian, cmap='gray')
                ax2.imshow(magnitude, cmap='gray')
                ax3.imshow(nms, cmap='gray')
                ax4.imshow(after_threshold, cmap='gray')
                ax5.imshow(output, cmap='gray')
                ax1.set_title('Gaussian Image')
                ax2.set_title('Gradient')
                ax3.set_title('NMS')
                ax4.set_title('Thresholding')
                ax5.set_title('Edge Linking')
                plt.suptitle('Results: Filter size={}, Sigma={}, Percentage of non-edges={}'.format(n, s, p))
                #save image
                plt.savefig(str(img_name) + '_image_result/Results_filter_size_' + str(n) + "sigma_val_" + str(s) + "percentage_of_non_edge_" + str(p) + kernel_name + ".png")







