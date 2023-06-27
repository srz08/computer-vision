import cv2
import csv
import os

# img = cv2.imread('pointer1.bmp')

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Get skin pixel at (x, y)
#         skin_pixel = img[y, x]
#         print('Skin pixel:', skin_pixel)

#         # Write skin pixel to CSV file
#         with open('skin_pixels.csv', 'a', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(skin_pixel)

# cv2.namedWindow('Image')
# cv2.setMouseCallback('Image', mouse_callback)

# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#loop over files in directiry
training_images = []
directory = "/Users/simonzouki/Documents/Northwestern/Q3/CV/MP4/training"
for file in os.listdir(directory):
    if file.endswith(".png"):
        training_images.append(os.path.join(directory, file))
        continue
    else:
        continue


#loop over images in list
for image in training_images:
    #put all pixels in csv file
    img = cv2.imread(image)
    with open('skin_pixels.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                writer.writerow(img[i, j])
