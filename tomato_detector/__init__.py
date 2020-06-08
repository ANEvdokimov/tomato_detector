import cv2
from tomato_detector import iem
from tomato_detector import color_filter
from tomato_detector import form_filter

DEFAULT_COLOR_SETTINGS = {"red": {"min": (0, 132, 0), "max": (179, 255, 255)}}


def detect_in_image(image, color_ranges_hsv=None, equalize_light=True, color_balance=True, split_tomatoes=True,
                    simple_blob_detector_params=None):
    if color_ranges_hsv is None:
        color_ranges_hsv = DEFAULT_COLOR_SETTINGS

    if equalize_light:
        image = iem.equalize_light(image)
    if color_balance:
        image = iem.white_balance(image)

    image_source_blur = cv2.bilateralFilter(image, 5, 75, 75)

    if simple_blob_detector_params is None:
        simple_blob_detector_params = form_filter.get_parameters_for_single_tomato(image.shape[0], image.shape[1])
    blob_detector = cv2.SimpleBlobDetector_create(simple_blob_detector_params)

    masks_tomatoes = {}
    tomato_keypoints = {}
    for name, color_range in color_ranges_hsv.items():
        mask_tomatoes = color_filter.find_tomatoes_by_color(image_source_blur, color_range)
        mask_tomatoes_processed = cv2.morphologyEx(mask_tomatoes, cv2.MORPH_OPEN, (50, 50), iterations=5)
        tomato_keypoints[name], masks_tomatoes[name] = form_filter.find_tomatoes(image_source_blur, mask_tomatoes_processed,
                                                                                 blob_detector, split_tomatoes)

    return tomato_keypoints, masks_tomatoes
