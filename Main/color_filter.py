import cv2


def find_tomatoes_by_color(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_Lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    # first condition: b*<138 => background
    mask_background = cv2.inRange(image_Lab, (0, 1, 1), (255, 255, 138))
    mask_without_background = cv2.bitwise_not(mask_background)

    # second condition: r>=72 => without leaf
    image_bgr_without_background = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_without_background)
    mask_without_background_leafs = cv2.inRange(image_bgr_without_background, (0, 0, 72), (255, 255, 255))

    # third condition: S>=132 => fruit
    image_hsv_without_background_and_leafs = cv2.bitwise_and(image_hsv, image_hsv, mask=mask_without_background_leafs)
    mask_red_tomatoes = cv2.inRange(image_hsv_without_background_and_leafs, (0, 132, 0), (179, 255, 255))
    # mask_without_background_leafs_fruit1 = cv2.subtract(mask_without_background_leafs, mask_red_tomatoes)

    return mask_red_tomatoes
