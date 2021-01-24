# Copyright Ary Noviyanto 2021
# Reference: 
#   Noviyanto, A., & Arymurthy, A. M. (2013). 
#   Beef cattle identification based on muzzle pattern using a matching 
#   refinement technique in the SIFT method. Computers and electronics 
#   in agriculture, 99, 77-84.
# 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

def get_refined_matches(matches, kp_1, kp_2):
    matchedKeypoints = []

    n = len(matches)
    if (n < 1):
        return None

    diff = np.zeros(n)
    num_bin_percentage = 0.1
    nbin = int(np.ceil(num_bin_percentage * n))
    truth_bin = []

    t = 4  # maximum different orientation in degree
    agreement_rate = 0.5 # 50% from all matched keypoints

    # generate orientation difference array
    for i in range(n):
        key_point_1 = kp_1[matches[i][0].queryIdx]
        key_point_2 = kp_2[matches[i][0].trainIdx]
        diff[i] = np.abs(key_point_1.angle - key_point_2.angle)
        if (diff[i] > 180.0):
            diff[i] = 360.0 - diff[i]

    for i in range(n):
        if (len(truth_bin) < nbin):
            numT = 0
            for j in range(n):
                if (j != i):
                    if (np.abs(diff[j]-diff[i]) < t):
                        numT = numT + 1
                        if (numT >= agreement_rate * n):
                            break

            if (numT >= agreement_rate * n):
                truth_bin.append(diff[i])
                # matchedKeypoints.append(matches[i])

        else:
            flag = True
            for j in range(nbin):
                if (flag):
                    flag = flag and (np.abs(truth_bin[j]-diff[i]) < t)
                else:
                    break

            if (flag):
                matchedKeypoints.append(matches[i])
        
    return matchedKeypoints

def get_matched_keypoints(filename_1, filename_2, dir='images'):

    file_path_1 = os.path.join(dir, filename_1)
    img_1 = cv2.imread(file_path_1)
    file_path_2 = os.path.join(dir, filename_2)
    img_2 = cv2.imread(file_path_2)

    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp_1, des_1 = sift.detectAndCompute(gray_1, None)
    kp_2, des_2 = sift.detectAndCompute(gray_2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_1, des_2, k=2)

    ratio_treshold = 0.75 # 75% between first and second matched keypoints' distances
    valid_matches = []
    for k_1, k_2 in matches:
        if k_1.distance < ratio_treshold * k_2.distance:
            valid_matches.append([k_1])

    return valid_matches, img_1, img_2, kp_1, kp_2

if __name__ == '__main__':
    filename_1 = sys.argv[1]
    filename_2 = sys.argv[2]

    valid_matches, img_1, img_2, kp_1, kp_2 = get_matched_keypoints(filename_1, filename_2)

    matched_img = cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, valid_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_img)
    plt.title('Matched keypoints:' + str(len(valid_matches)))
    plt.show()

    refiened_valid_matches = get_refined_matches(valid_matches, kp_1, kp_2)
    matched_img = cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, refiened_valid_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_img)
    plt.title('Matched keypoints after refinement: ' + str(len(refiened_valid_matches)))
    plt.show()