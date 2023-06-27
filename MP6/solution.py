import cv2
import numpy as np

def hough_transform(image, threshold=10):
    rho = 1
    theta = np.pi / 180
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough transform on the edge detected image manually
    lines = []
    for x in range(edges.shape[0]):
        for y in range(edges.shape[1]):
            if edges[x][y] == 255:
                for t in range(0, 180):
                    r = int(x * np.cos(t * theta) + y * np.sin(t * theta))
                    lines.append([r, t])
    
    lines = np.array(lin


    # Convert the (rho, theta) values to cartesian coordinates
    if lines is not None:
        lines = lines[:, 0, :]
        x1 = np.cos(lines[:, 1]) * lines[:, 0]
        y1 = np.sin(lines[:, 1]) * lines[:, 0]
        x2 = np.cos(lines[:, 1] + np.pi) * lines[:, 0]
        y2 = np.sin(lines[:, 1] + np.pi) * lines[:, 0]
        lines = np.vstack((x1, y1, x2, y2)).T

        # Find the most prominent lines (those with a vote count above the threshold)
        peaks = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=20, maxLineGap=10)
        if peaks is not None:
            peaks = peaks[:, 0, :]

    return lines, peaks

image = cv2.imread('test.bmp')
lines, peaks = hough_transform(image)
#plot the lines on new image
for x1,y1,x2,y2 in peaks:
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()