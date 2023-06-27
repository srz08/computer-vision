import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def get_files(directory):
    files = os.listdir(directory)
    files.sort()
    return files

def generate_video(base_img, input_directory, output_video_path):
    w = base_img.shape[1]
    h = base_img.shape[0]
    video = cv2.VideoWriter('{}.mp4'.format(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), 40, (w, h))

    prev_point = (52.5, 25)
    limit = 15

    files = get_files(input_directory)
    for file in files:
        print(file)
        img_name = "{}".format(file)
        new_img = cv2.imread(input_directory + '/{}'.format(img_name))
        video.write(new_img)

    cv2.destroyAllWindows()
    video.release()

base_img = cv2.imread('/Users/simonzouki/Documents/Northwestern/Q3/CV/project/main_folder/results/run4/frame_0000.jpg')
generate_video(base_img, "/Users/simonzouki/Documents/Northwestern/Q3/CV/project/main_folder/results/run4", "ssd")