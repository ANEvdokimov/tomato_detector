import cv2
import numpy as np


def get_parameters_for_single_tomato(frame_height, frame_width):
    parameters = cv2.SimpleBlobDetector_Params()

    parameters.filterByColor = False

    parameters.filterByArea = True
    parameters.minArea = frame_height/10 * frame_width/10
    parameters.maxArea = frame_height/3 * frame_width/3

    parameters.filterByCircularity = True
    parameters.minCircularity = 0.6

    parameters.filterByConvexity = True
    parameters.minConvexity = 0.7

    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.7

    return parameters


def detect(detector_parameters, image):
    temp = cv2.SimpleBlobDetector_create(detector_parameters).detect(image)
    return temp


def draw_keypoints(image, keypoints, color):
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    return cv2.drawKeypoints(image, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
