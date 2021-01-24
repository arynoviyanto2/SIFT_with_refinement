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

def get_refined_matches(matches, kp_1, kp_2):
    matchedKeypoints = []

    n = len(matches)
    diff = np.zeros(n)

    nbin = 5
    truth_bin = []

    t = 12 # maximum different orientation in degree
    agreement_rate = 0.5 # 50% from all matched keypoints

    # generate orientation difference array
    for i in range(n):
        key_point_1 = kp_1[matches[i][0].queryIdx]
        key_point_2 = kp_2[matches[i][0].trainIdx]
        diff[i] = np.abs(key_point_1.angle - key_point_2.angle)
        if (diff[i] > 180.0):
            diff[i] = 360.0 - diff[i]

    for i in range(n):
        if (len(truth_bin) < 5):
            numT = 0
            for j in range(n):
                if (j != i):
                    if (np.abs(diff[j]-diff[i]) < t):
                        numT = numT + 1
                        if (numT >= agreement_rate * n):
                            break

            if (numT >= agreement_rate * n):
                truth_bin.append(diff[i])
                matchedKeypoints.append(matches[i])

        else:
            flag = True
            for j in range(nbin):
                if (flag):
                    flag = flag and (abs(truth_bin[j]-diff[i]) < t)
                else:
                    break

            if (flag):
                matchedKeypoints.append(matches[i])
        
    return matchedKeypoints


img_1 = cv2.imread('images/23_0001.jpg')
#img_2 = cv2.imread('images/23_0002.jpg')
img_2 = cv2.imread('images/24_0012.jpg')

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
    if k_1.distance < 0.75 * k_2.distance:
        valid_matches.append([k_1])

matched_img = cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, valid_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matched_img)
plt.title('Matched keypoints')
plt.show()

refiened_valid_matches = get_refined_matches(valid_matches, kp_1, kp_2)
matched_img = cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, refiened_valid_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matched_img)
plt.title('Matched keypoints after refinement')
plt.show()
