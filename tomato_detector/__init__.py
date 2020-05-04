import cv2
from tomato_detector import iem
from tomato_detector import color_filter
from tomato_detector import hole_filler
from tomato_detector import form_filter


DEFAULT_COLOR_SETTINGS = {"red": {"min": (0, 132, 0), "max": (179, 255, 255)}}


def detect_in_image(image, color_ranges_hsv=None, equalize_light=True):
    if color_ranges_hsv is None:
        color_ranges_hsv = DEFAULT_COLOR_SETTINGS

    if equalize_light:
        image = iem.equalize_light(image)

    image_source_blur = cv2.GaussianBlur(image, (5, 5), 0)

    masks_tomatoes = {}
    for name, color_range in color_ranges_hsv.items():
        masks_tomatoes[name] = color_filter.find_tomatoes_by_color(image_source_blur, color_range)

    masks_tomatoes_processed = {}
    for name, mask in masks_tomatoes.items():
        mask = cv2.erode(mask, (30, 30), iterations=5)
        # mask_fruit_dilate = cv2.dilate(mask_fruit_erode, (5, 5), iterations=2)
        masks_tomatoes_processed[name] = mask

    tomato_keypoints = {}
    for name, mask in masks_tomatoes_processed.items():
        mask_without_holes = hole_filler.fill(cv2.bitwise_and(image, image, mask=mask), mask)
        tomato_keypoints[name] = form_filter.find_tomatoes(mask_without_holes)

    return tomato_keypoints, masks_tomatoes_processed
