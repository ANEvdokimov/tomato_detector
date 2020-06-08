from tomato_detector import hole_filler
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2


def find_tomatoes(image_bgr, mask, blob_detector, split_tomatoes):
    if split_tomatoes:
        mask_copy = hole_filler.fill(mask)

        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        distance_map = ndimage.distance_transform_edt(mask_copy)

        min_distance = mask.shape[0]/10
        local_max = peak_local_max(distance_map, indices=False, min_distance=np.uint8(min_distance), labels=mask_copy)

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance_map, markers, mask=mask_copy)

        # loop over the unique labels returned by the Watershed
        # algorithm
        keypoints = []
        masks_tomatoes = []
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue

            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask_with_one_object = np.zeros(mask_copy.shape, dtype="uint8")
            mask_with_one_object[labels == label] = 255

            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            image_with_one_tomato = cv2.bitwise_and(image_gray, image_gray, mask=mask_with_one_object)
            keypoint = blob_detector.detect(image_with_one_tomato)
            if len(keypoint) != 0:
                keypoints = keypoints + keypoint
                masks_tomatoes.append(mask_with_one_object)

        return keypoints, masks_tomatoes
    else:
        keypoints = blob_detector.detect(mask)
        return keypoints, mask


def get_parameters_for_single_tomato(frame_height, frame_width):
    parameters = cv2.SimpleBlobDetector_Params()

    parameters.filterByColor = False

    parameters.filterByArea = True
    parameters.minArea = frame_height / 15 * frame_width / 15
    parameters.maxArea = frame_height / 1.5 * frame_width / 1.5

    parameters.filterByCircularity = True
    parameters.minCircularity = 0.6

    parameters.filterByConvexity = True
    parameters.minConvexity = 0.5

    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.4

    return parameters
