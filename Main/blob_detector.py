import cv2
import numpy as np


def detect(image):
    height, width = image.shape[:2]

    # Setup SimpleBlobDetector parameters.
    parameters = cv2.SimpleBlobDetector_Params()

    parameters.filterByColor = False

    parameters.filterByArea = True
    parameters.minArea = height/10 * width/10
    parameters.maxArea = height/3 * width/3

    parameters.filterByCircularity = True
    parameters.minCircularity = 0.6

    parameters.filterByConvexity = True
    parameters.minConvexity = 0.5

    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.5

    # Set up the detector with the parameters.
    detector = cv2.SimpleBlobDetector_create(parameters)

    # Detect blobs.
    return detector.detect(image)


def draw_keypoints(image, keypoints):
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    return cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
