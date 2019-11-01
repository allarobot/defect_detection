#!/usr/bin/python
"""
monitor blob variation with a reference points data
this is a runtime operation code
"""


import cv2
import numpy as np
import pickle
import blob

fileName = "data/BlobTest2b.jpg"
kpt = "keypoints"

# Load blobs points
file = open(kpt, "rb")
pnts_ref = pickle.load(file)
file.close()


def monitor(pnts1, pnts2):
    '''
    check the appearance of reference key points
    :param pnts1: reference key points
    :param pnts2: realtime detection key points
    :return: key points which are disappeared
    '''
    keypoints = []
    for pt, size in pnts1:
        keypoint = cv2.KeyPoint(pt[0], pt[1], size)
        overlap = False
        for pnt in pnts2:
            val = cv2.KeyPoint_overlap(pnt,keypoint)
            if val > 0.5:
                overlap = True
                continue
        if not overlap:
            keypoints.append(keypoint)
    return keypoints


# Read image
im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

# Detect blobs.
keypoints,im_with_keypoints = blob.detection(im)

#
pnts = monitor(pnts_ref,keypoints)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, pnts, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)