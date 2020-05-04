import cv2
import copy


def fill(image, binary_mask):
    binary_mask_copy = copy.copy(binary_mask)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary_mask_copy, contours, -1, 255, cv2.FILLED)
    return binary_mask_copy
