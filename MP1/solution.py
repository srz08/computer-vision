from PIL import Image
import numpy as np

def connected_components_row_by_row(image_path, threshold=32):
    #open bmp file
    image = Image.open(image_path)
    #loop through each pixel
    matrix = [[image.getpixel((x,y)) for x in range(image.size[0])] for y in range(image.size[1])]
    conflicts = []
    count = 0
    for x in range(len(matrix)):
        for y in range(len(matrix[0])): 
            if matrix[x][y] == 255:
                left = matrix[x-1][y] if x > 0 else 0
                top = matrix[x][y-1] if y > 0 else 0
                if left == 0 and top == 0:
                    count += 1
                    matrix[x][y] = count
                elif left == top:
                    matrix[x][y] = left
                elif left == 0 and top != 0:
                    matrix[x][y] = top
                elif left != 0 and top == 0:
                    matrix[x][y] = left
                else:
                    matrix[x][y] = top
                    conflicts.append((left, top))

    unique_values = {}
    #fix the conflicts
    for x in range(len(matrix)):
        for y in range(len(matrix[0])): 
            if matrix[x][y] != 0:
                for conflict in conflicts:
                    if matrix[x][y] == conflict[0]:
                        matrix[x][y] = conflict[1]
                if matrix[x][y] not in unique_values:
                    unique_values[matrix[x][y]] = 1
                else:
                    unique_values[matrix[x][y]] += 1

    for x in range(len(matrix)):
        for y in range(len(matrix[0])): 
            if matrix[x][y] != 0 and unique_values[matrix[x][y]] < threshold:
                matrix[x][y] = 0

    #delete values in unique_values that are less than threshold
    for key in list(unique_values):
        if unique_values[key] < threshold:
            del unique_values[key]

    #use map_labels_to_colors to get the colors
    map_labels_to_colors(matrix, unique_values, image_path)
    
    return matrix, len(unique_values)

def map_labels_to_colors(matrix, labels, image_path):
    legth = len(labels)
    colors = {}
    start = 100
    end = 250
    step = int((end - start) / legth)
    for label in labels:
        colors[label] = start
        start += step
    
    #loop over matrix and change the values to the colors
    for x in range(len(matrix)):
        for y in range(len(matrix[0])): 
            if matrix[x][y] != 0:
                matrix[x][y] = colors[matrix[x][y]]

    #turn matrx to np array
    matrix = np.array(matrix, dtype=np.int64)
    #save the image
    img = Image.fromarray(matrix.astype(np.uint8))
    img.save(image_path + "_output.bmp")

             
image_paths = ["face.bmp", "face_old.bmp", "gun.bmp"]
for image_path in image_paths:
    matrix, length = connected_components_row_by_row(image_path)
    print("For {}, we have {} connected component(s)".format(image_path, length))
