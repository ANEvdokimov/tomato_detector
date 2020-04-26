import cv2


def equalize_light(image):
    clahe = cv2.createCLAHE(clipLimit=3)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)

    l_el = clahe.apply(l)
    image_lab_el = cv2.merge((l_el, a, b))

    return cv2.cvtColor(image_lab_el, cv2.COLOR_LAB2BGR)
