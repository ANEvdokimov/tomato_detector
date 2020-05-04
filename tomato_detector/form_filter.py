from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2
import copy


def find_tomatoes(mask):
    mask_copy = copy.copy(mask)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(mask_copy)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=mask_copy)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=mask_copy)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    parameters = get_parameters_for_single_tomato(mask_copy.shape[0], mask_copy.shape[1])

    # loop over the unique labels returned by the Watershed
    # algorithm
    keypoints = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask_with_one_object = np.zeros(mask_copy.shape, dtype="uint8")
        mask_with_one_object[labels == label] = 255

        keypoint = detect(parameters, mask_with_one_object)
        if len(keypoint) != 0:
            keypoints = keypoints + keypoint

    return keypoints


def get_parameters_for_single_tomato(frame_height, frame_width):
    parameters = cv2.SimpleBlobDetector_Params()

    parameters.filterByColor = False

    parameters.filterByArea = False
    parameters.minArea = frame_height/10 * frame_width/10
    parameters.maxArea = frame_height/3 * frame_width/3

    parameters.filterByCircularity = True
    parameters.minCircularity = 0.5

    parameters.filterByConvexity = True
    parameters.minConvexity = 0.5

    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.5

    return parameters


def detect(detector_parameters, image):
    return cv2.SimpleBlobDetector_create(detector_parameters).detect(image)


def draw_keypoints(image, keypoints, color_bgr):
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    return cv2.drawKeypoints(image, keypoints, np.array([]), color_bgr, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
