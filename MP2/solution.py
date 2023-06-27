from PIL import Image
import numpy as np

def image_to_matrix(image_path):
    image = Image.open(image_path)
    matrix = [[image.getpixel((x, y)) for x in range(image.width)] for y in range(image.height)]
    return matrix

def check_erose(matrix, x, y, filter):
    mid = len(filter) // 2
    for i in range(len(filter)):
        for j in range(len(filter)):
            if filter[i][j] == 1:
                if matrix[x + i - mid][y + j - mid] == 0:
                    return False
    return True

def fill_dilate(matrix, x, y, filter):
    mid = len(filter) // 2
    for i in range(len(filter)):
        for j in range(len(filter)):
            if filter[i][j] == 1:
                matrix[x + i - mid][y + j - mid] = 255

def dilate(matrix, filter):
    #add padding
    new_matrix = [[0 for x in range(len(matrix[0]) + len(filter) - 1)] for y in range(len(matrix) + len(filter) - 1)]
    # new_matrix = [[0 for x in range(len(matrix[0]))] for y in range(len(matrix))]
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if matrix[x][y] == 255:
                fill_dilate(new_matrix, x, y, filter)
    return new_matrix


def erose(matrix, filter):
    #add padding
    new_matrix = [[0 for x in range(len(matrix[0]) + len(filter) - 1)] for y in range(len(matrix) + len(filter) - 1)]
    # new_matrix = [[0 for x in range(len(matrix[0]))] for y in range(len(matrix))]
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if check_erose(matrix, x, y, filter):
                new_matrix[x][y] = 255
    return new_matrix

def opening(matrix, filter):
    new_matrix = erose(matrix, filter)
    new_matrix = dilate(new_matrix, filter)
    return new_matrix

def closing(matrix, filter):
    new_matrix = dilate(matrix, filter)
    new_matrix = erose(new_matrix, filter)
    return new_matrix

def boundary(matrix, filter):
    new_matrix = [[0 for x in range(len(matrix[0]))] for y in range(len(matrix))]
    erose_matrix = erose(matrix, filter)
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            new_matrix[x][y] = matrix[x][y] - erose_matrix[x][y]
    return new_matrix

def clear_boundary(matrix, closing_filter, boundary_filter):
    matrix = closing(matrix, closing_filter)
    matrix = boundary(matrix, boundary_filter)
    return matrix

image_path = 'palm.bmp'
matrix = image_to_matrix(image_path)


filter_1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
filter_2 = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
# filter = [[1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1],  [1,1,1,1,1,1,1]]


functions = [erose, dilate, opening, closing, boundary]
for function in functions:
    new_matrix = function(matrix, filter)
    new_matrix = np.array(new_matrix, dtype=np.int64)
    img = Image.fromarray(new_matrix.astype(np.uint8))
    img.save("{}/".format(function.__name__) + "4_" + image_path)


# new_matrix = clear_boundary(matrix, filter_2, filter_1)
# new_matrix = np.array(new_matrix, dtype=np.int64)
# img = Image.fromarray(new_matrix.astype(np.uint8))
# img.save("{}/first_".format(clear_boundary.__name__) + image_path)