import cv2
import copy


def fill(image, binary_mask):
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # mask = np.zeros((binary_mask.shape[0] + 2, binary_mask.shape[1] + 2), np.uint8)
    # temp = cv2.floodFill(binary_mask, mask, (0, 0), 255, flags=cv2.FLOODFILL_MASK_ONLY)
    # return binary_mask

    binary_mask_copy = copy.copy(binary_mask)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary_mask_copy, contours, -1, 255, cv2.FILLED)
    return binary_mask_copy
