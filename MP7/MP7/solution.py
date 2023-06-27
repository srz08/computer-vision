import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



def crop_image(img, p1, p2, p3, p4):
    crop_img = img[int(p2[1]):int(p1[1]), int(p2[0]):int(p3[0])]
    return crop_img

def get_files(directory):
    files = os.listdir(directory)
    files.sort()
    return files

def ssd(img1, img2):
    return np.sum((img1 - img2)**2)

def normalized_cross_correlation(img1, img2):
    avg1 = np.mean(img1)
    avg2 = np.mean(img2)
    norm_cross_correl = np.sum((img1 - avg1) * (img2 - avg2)) / (np.sqrt(np.sum((img1 - avg1)**2)) * np.sqrt(np.sum((img2 - avg2)**2)))
    return norm_cross_correl

def cross_correlation(img1, img2):
    cross_correl = np.sum((img1 * img2))
    return cross_correl

def generate_video(base_img, crop_img, cover_img, side_img, side_img_2, side_img_3, back_head_img, input_directory, output_video_path, method):
    w = base_img.shape[1]
    h = base_img.shape[0]
    video = cv2.VideoWriter('{}.mp4'.format(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), 40, (w, h))

    prev_point = (52.5, 25)
    limit = 15

    files = get_files(input_directory)
    for file in files:
        print(file)
        img_name = "{}".format(file)
        new_img = cv2.imread('image_girl/{}'.format(img_name))
        bounding_box_height = crop_img.shape[0]
        bounding_box_width = crop_img.shape[1]


        ssd_dic = {}
        cross_correlation_dic = {}
        normalized_cross_correlation_dic = {}
        for i in range(new_img.shape[0] - bounding_box_height):
            for j in range(new_img.shape[1] - bounding_box_width):
                if abs(i - prev_point[1]) > limit or abs(j - prev_point[0]) > limit:
                    continue
                new_img_crop = new_img[i:i+bounding_box_height, j:j+bounding_box_width]
                if method == "ssd":
                    face_ssd = ssd(new_img_crop, crop_img)
                    back_ssd = ssd(new_img_crop, back_head_img)
                    side_ssd = ssd(new_img_crop, side_img)
                    side_ssd_2 = ssd(new_img_crop, side_img)
                    side_ssd_3 = ssd(new_img_crop, side_img)
                    ssd_dic[(i, j)] = max(back_ssd, face_ssd, side_ssd, side_ssd_2, side_ssd_3)
                if method == "cross_correlation":
                    face_cross_corr = cross_correlation(new_img_crop, crop_img)
                    cover_img_corr = cross_correlation(new_img_crop, cover_img)
                    side_cross_corr = cross_correlation(new_img_crop, side_img)
                    side_cross_corr_2 = cross_correlation(new_img_crop, side_img_2)
                    side_cross_corr_3 = cross_correlation(new_img_crop, side_img_3)
                    cross_correlation_dic[(i, j)] = max(cover_img_corr, face_cross_corr, side_cross_corr, side_cross_corr_2, side_cross_corr_3)
                if method == "normalized_cross_correlation":
                    face_norm_cross = normalized_cross_correlation(new_img_crop, crop_img)
                    cover_img_cross = normalized_cross_correlation(new_img_crop, cover_img)
                    side_norm_cross = normalized_cross_correlation(new_img_crop, side_img)
                    side_norm_cross_2 = normalized_cross_correlation(new_img_crop, side_img_2)
                    side_norm_cross_3 = normalized_cross_correlation(new_img_crop, side_img_3)
                    normalized_cross_correlation_dic[(i, j)] = max(cover_img_cross, face_norm_cross, side_norm_cross, side_norm_cross_2, side_norm_cross_3)

        if method == "ssd":
            min_ssd = min(ssd_dic.values())
            min_ssd_key = [k for k, v in ssd_dic.items() if v == min_ssd][0]
        elif method == "cross_correlation":
            max_correlation = max(cross_correlation_dic.values())
            max_correlation_key = [k for k, v in cross_correlation_dic.items() if v == max_correlation][0]
        elif method == "normalized_cross_correlation":
            max_normalized_correlation = max(normalized_cross_correlation_dic.values())
            max_normalized_correlation_key = [k for k, v in normalized_cross_correlation_dic.items() if v == max_normalized_correlation][0]
        if method == "ssd":
            i = min_ssd_key[0]
            j = min_ssd_key[1]
        elif method == "cross_correlation":
            i = max_correlation_key[0]
            j = max_correlation_key[1]
        elif method == "normalized_cross_correlation":
            i = max_normalized_correlation_key[0]
            j = max_normalized_correlation_key[1]
        
        cv2.rectangle(new_img, (j, i), (j+bounding_box_width, i+bounding_box_height), (0, 0, 255), 2)

        video.write(new_img)

        prev_point = (j, i)

    cv2.destroyAllWindows()
    video.release()

base_img = cv2.imread('image_girl/0001.jpg')
p1, p2, p3, p4 = (52.5, 65), (52.5, 25), (90, 25), (90, 65)
crop_img = crop_image(base_img, p1, p2, p3, p4)


w, h = crop_img.shape[1], crop_img.shape[0]
back_head_img = cv2.imread('image_girl/0200.jpg')
left_corner = (43, 29)
back_head_img = back_head_img[int(left_corner[1]):int(left_corner[1])+h, int(left_corner[0]):int(left_corner[0])+w]

side_img = cv2.imread('image_girl/0085.jpg')
left_corner = (45, 30)
side_img = side_img[int(left_corner[1]):int(left_corner[1])+h, int(left_corner[0]):int(left_corner[0])+w]

side_img_2 = cv2.imread('image_girl/0308.jpg')
left_corner = (54, 32)
side_img_2 = side_img_2[int(left_corner[1]):int(left_corner[1])+h, int(left_corner[0]):int(left_corner[0])+w]

side_img_3 = cv2.imread('image_girl/0330.jpg')
left_corner = (41, 31)
side_img_3 = side_img_3[int(left_corner[1]):int(left_corner[1])+h, int(left_corner[0]):int(left_corner[0])+w]

cover_img = cv2.imread('image_girl/0458.jpg')
left_corner = (48, 26)
cover_img = cover_img[int(left_corner[1]):int(left_corner[1])+h, int(left_corner[0]):int(left_corner[0])+w]

generate_video(base_img, crop_img, cover_img, side_img, side_img_2, side_img_3, back_head_img, "./image_girl", "ssd", "ssd")



