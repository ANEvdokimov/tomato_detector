import cv2
from Main import iem
from Main import color_filter
from Main import hole_filler
from Main import tomato_detector


vid = cv2.VideoCapture("../Dataset/2_720.mp4")

while vid.isOpened():
    _, image_source = vid.read()
    if image_source is None:
        break

    image_source_equalized_light = iem.equalize_light(image_source)
    image_source_blur = cv2.GaussianBlur(image_source_equalized_light, (5, 5), 0)
    # cv2.imshow("wb", image_source_blur)

    mask_red_tomatoes = color_filter.find_tomatoes_by_color(image_source_blur)

    mask_fruit_erode = cv2.erode(mask_red_tomatoes, (30, 30), iterations=5)
    # mask_fruit_dilate = cv2.dilate(mask_fruit_erode, (5, 5), iterations=2)
    mask_fruit_dilate = mask_fruit_erode
    # cv2.imshow("tomatoes", cv2.bitwise_and(image_source, image_source, mask=mask_fruit_dilate))

    new_mask = hole_filler.fill(cv2.bitwise_and(image_source, image_source, mask=mask_fruit_dilate), mask_fruit_dilate)
    keypoints = tomato_detector.segmentation(new_mask)
    image_with_circles = tomato_detector.draw_keypoints(image_source, keypoints, (0, 255, 0))

    cv2.imshow('detected circles', image_with_circles)
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
