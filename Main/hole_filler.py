import cv2
import numpy as np
import copy


def fill(binary_mask):
    # Copy the thresholded image.
    binary_mask_copy = copy.copy(binary_mask)

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = binary_mask_copy.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(binary_mask_copy, mask, (0, 0), 255)

    # Invert floodfilled image
    binary_mask_copy_inv = cv2.bitwise_not(binary_mask_copy)

    # Combine the two images to get the foreground.
    return binary_mask | binary_mask_copy_inv
