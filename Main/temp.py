# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import copy


def segmentation(image_source, thresh):
    thresh_copy = copy.copy(thresh)
    # # construct the argument parse and parse the arguments
    # # ap = argparse.ArgumentParser()
    # # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # # args = vars(ap.parse_args())
    #
    #
    # # load the image and perform pyramid mean shift filtering
    # # to aid the thresholding step
    # image = cv2.imread("c:/Users/evdok/Desktop/watershed_coins_01.jpg")
    # shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    # cv2.imshow("Input", image)
    #
    # # convert the mean shift image to grayscale, then apply
    # # Otsu's thresholding
    # gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow("Thresh", thresh)


    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)


    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    parameters = get_parameters_for_single_tomato(thresh.shape[0], thresh.shape[1])

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
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        # ((x, y), r) = cv2.minEnclosingCircle(c)
        # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        keypoint = detect(parameters, mask)
        if len(keypoint) != 0:
            keypoints = keypoints + keypoint

    image = draw_keypoints(thresh_copy, keypoints, (0, 255, 0))

    # show the output image
    return image


def get_parameters_for_single_tomato(frame_height, frame_width):
    parameters = cv2.SimpleBlobDetector_Params()

    parameters.filterByColor = False

    parameters.filterByArea = True
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


def draw_keypoints(image, keypoints, color):
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    return cv2.drawKeypoints(image, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
