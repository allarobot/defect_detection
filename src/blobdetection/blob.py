#!/usr/bin/python
"""
refer to
https://www.learnopencv.com/blob-detection-using-opencv-python-c/


"""

import cv2
import numpy as np


def createDetector():
    '''
    customized blob detector
    :return: detector
    '''
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # # Set up the detector with default parameters.
    # detector = cv2.SimpleBlobDetector()
    return detector


def detection(im):
    '''
    shining blob detection
    reversing image by 255-image

    :param im:
    :return: keypoints, image with points
    '''
    # Detector
    detector = createDetector()

    im = 255-im
    # Detect blobs.
    keypoints = detector.detect(im)


    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, im_with_keypoints


if __name__ == "__main__":
    # Image file name
    fileName = "data/BlobTest.jpg"

    # Read image
    im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    # im2 = 255-im
    # cv2.imwrite("data/BlobTest2.jpg",im2)

    keypoints,im_with_keypoints = detection(im)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)