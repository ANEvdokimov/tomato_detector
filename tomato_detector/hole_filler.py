import cv2
import copy


def fill(binary_mask):
    """
    Функия заполнения внутренних отверстий объектов на бинаризованном изображении

    binary_mask -- бинаризованное изображение
    """
    binary_mask_copy = copy.copy(binary_mask)
    contours, hierarchy = cv2.findContours(binary_mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary_mask_copy, contours, -1, 255, cv2.FILLED)
    return binary_mask_copy
