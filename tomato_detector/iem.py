import cv2
import numpy as np


def equalize_light(image):
    """
    Выравнивание яркости bgr-изображения для повышения контраста
    Equation the brightness of a bgr-image to enhance contrast

    image -- an BGR-image
    """
    clahe = cv2.createCLAHE(clipLimit=3)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)

    l_el = clahe.apply(l)
    image_lab_el = cv2.merge((l_el, a, b))
    return cv2.cvtColor(image_lab_el, cv2.COLOR_LAB2BGR)


def white_balance(image):
    """
    Корректировка баланса юелого изображения (Gray World)
    Adjusting the image white balance (Gray World)

    image -- an BGR-image
    """
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(image_lab[:, :, 1])
    avg_b = np.average(image_lab[:, :, 2])
    image_lab[:, :, 1] = image_lab[:, :, 1] - ((avg_a - 128) * (image_lab[:, :, 0] / 255.0) * 1.1)
    image_lab[:, :, 2] = image_lab[:, :, 2] - ((avg_b - 128) * (image_lab[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
