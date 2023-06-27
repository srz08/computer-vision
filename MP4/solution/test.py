import numpy as np
import cv2
import os


hist_path = 'histogram/skin_color_histogram_16.csv'
img_path = 'test/Hand_0000003.jpg'
def test(img_path, hist_path, close=True, open=True):
    hist = np.genfromtxt(hist_path, delimiter=',')
    img = cv2.imread(img_path)
    img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h_channel = img_hsi[:, :, 0] / 180.0
    s_channel = img_hsi[:, :, 1] / 255.0

    threshold = 0.001
    count = 0
    skin_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h = h_channel[i, j]
            s = s_channel[i, j]
            skin_hist = hist[int(h * (hist.shape[0]-1)), int(s * (hist.shape[1]-1))]
            if skin_hist >= threshold:
                count += 1
                skin_mask[i, j] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if close:
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    if open:
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    masked_img = cv2.bitwise_and(img, img, mask=skin_mask)
    return masked_img

test_image = []
directory = "/Users/simonzouki/Documents/Northwestern/Q3/CV/MP4/test"
for file in os.listdir(directory):
    if file.endswith(".jpg") or file.endswith(".bmp") or file.endswith(".jpeg"):
        test_image.append(os.path.join(directory, file))
        continue
    else:
        continue

for image in test_image:
    masked_img = test(image, hist_path, close=True, open=True)
    cv2.imshow('test', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

