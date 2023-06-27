import numpy as np
import matplotlib.pyplot as plt
import colorsys

rgb_data = np.genfromtxt('skin_pixels.csv', delimiter=',')

rgb_data = rgb_data / 255.0

hsi_data = np.array([colorsys.rgb_to_hsv(r, g, b) for b, g, r in rgb_data])
print(np.mean(rgb_data, axis=0)*255.0)
h_channel = hsi_data[:, 0]
s_channel = hsi_data[:, 1]


num_bins = 32
h_range = [0, 1]
s_range = [0, 1]

hist, x_range, y_range = np.histogram2d(h_channel, s_channel, bins=[num_bins, num_bins], range=[h_range, s_range])
plt.imshow(hist, interpolation='nearest', extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]])

plt.show()

hist /= np.sum(hist)

np.savetxt('histogram/skin_color_histogram_32.csv', hist, delimiter=',')

