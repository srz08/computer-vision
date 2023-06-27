import numpy as np
import matplotlib.pyplot as plt
import colorsys
import cv2
import os

rgb_data = np.genfromtxt('skin_pixels.csv', delimiter=',')

rgb_data = rgb_data / 255.0

hsi_data = np.array([colorsys.rgb_to_hsv(r, g, b) for b, g, r in rgb_data])
print(np.mean(rgb_data, axis=0)*255.0)
h_channel = hsi_data[:, 0]
s_channel = hsi_data[:, 1]

#find the mean of the h and s channels
h_mean = np.mean(h_channel)
s_mean = np.mean(s_channel)

print(h_mean)
print(s_mean)

#find the standard deviation of the h and s channels
h_std = np.std(h_channel)
s_std = np.std(s_channel)

#run on test images
img_path = 'test/gun1.bmp'

def test(img_path, h_mean, h_std, s_mean, s_std):
    img = cv2.imread(img_path)
    img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nb_std = 2
    #loop over pixels
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h = img_hsi[i, j, 0] / 180.0
            s = img_hsi[i, j, 1] / 255.0
            if (h > h_mean - nb_std*h_std) and (h < h_mean + nb_std*h_std) and (s > s_mean - nb_std*s_std) and (s < s_mean + nb_std*s_std):
                img[i, j, :] = 255
            else:
                img[i, j, :] = 0

    return img

test_image = []
directory = "/Users/simonzouki/Documents/Northwestern/Q3/CV/MP4/test"
for file in os.listdir(directory):
    if file.endswith(".jpg") or file.endswith(".bmp") or file.endswith(".jpeg"):
        test_image.append(os.path.join(directory, file))
        continue
    else:
        continue

for image in test_image:
    masked_img = test(image, h_mean, h_std, s_mean, s_std)
    cv2.imshow('test', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




