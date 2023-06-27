from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def image_to_matrix(image_path):
    image = Image.open(image_path)
    matrix = [[image.getpixel((x, y)) for x in range(image.width)] for y in range(image.height)]
    return matrix

def rbg_to_grayscale(matrix):
    new_matrix = [[0 for x in range(len(matrix[0]))] for y in range(len(matrix))]
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            new_matrix[x][y] = (matrix[x][y][0] + matrix[x][y][1] + matrix[x][y][2]) // 3
    return new_matrix

def plot(histogram):
    plt.bar(range(len(histogram)), histogram)
    plt.show()

def matrix_to_histogram(matrix):
    histogram = [0 for x in range(256)]
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            histogram[int(matrix[x][y])] += 1
    return histogram

def histogram_equalization(matrix, histogram):
    cdf = [0 for x in range(256)]
    for i in range(len(histogram)):
        cdf[i] = cdf[i - 1] + histogram[i]

    last_val = cdf[-1]

    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            matrix[x][y] = (cdf[matrix[x][y]]/last_val) * 255
    return matrix, cdf


def linear_fit_to_plane(img):
    height, width = img.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    A = np.column_stack((X.ravel(), Y.ravel(), np.ones(height*width)))
    Z = img.ravel().reshape((-1,1))
    C = np.linalg.pinv(A) @ Z
    plane = (C[0]*X + C[1]*Y + C[2]).reshape((height, width))
    corrected_img = img - plane
    corrected_img = ((corrected_img - np.min(corrected_img)) / (np.max(corrected_img) - np.min(corrected_img))) * 255
    corrected_img = np.clip(corrected_img, 0, 255)
    corrected_img = Image.fromarray(corrected_img.astype('uint8'), mode='L')
    return corrected_img

def quadratic_fit_to_plane(img):
    height, width = img.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    A = np.column_stack((X.ravel() ** 2, Y.ravel() ** 2, X.ravel() * Y.ravel(), X.ravel(), Y.ravel(), np.ones(height*width)))
    Z = img.ravel().reshape((-1,1))
    C = np.linalg.pinv(A) @ Z
    plane = (C[0]*X**2 + C[1]*Y**2 + C[2]*X*Y + C[3]*X + C[4]*Y + C[5]).reshape((height, width))
    corrected_img = img - plane
    corrected_img = ((corrected_img - np.min(corrected_img)) / (np.max(corrected_img) - np.min(corrected_img))) * 255
    corrected_img = np.clip(corrected_img, 0, 255)
    corrected_img = Image.fromarray(corrected_img.astype('uint8'), mode='L')
    return corrected_img


def get_cdf(histogram):
    cdf = [0 for x in range(256)]
    for i in range(len(histogram)):
        cdf[i] = cdf[i - 1] + histogram[i]
    return cdf


image_path = 'moon.bmp'
matrix = image_to_matrix(image_path)
matrix = rbg_to_grayscale(matrix)

histogram = matrix_to_histogram(matrix)
# plot(histogram)
matrix, cdf = histogram_equalization(matrix, histogram)
# plot(cdf)
histogram = matrix_to_histogram(matrix)
# plot(histogram)
cdf = get_cdf(histogram)
# plot(cdf)


matrix = np.array(matrix, dtype=np.int64)
img = Image.fromarray(matrix.astype(np.uint8))
# img.save('moon_equalized.bmp')


corrected_image1 = linear_fit_to_plane(matrix)
# corrected_image.save('moon_corrected.bmp')

corrected_image2 = quadratic_fit_to_plane(matrix)
# corrected_image.save('moon_quadratic_corrected.bmp')
