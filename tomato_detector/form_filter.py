from tomato_detector import hole_filler
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import cv2


def find_tomatoes(image_bgr, mask, blob_detector, split_tomatoes=True, min_distance_when_splitting=0):
    """
    Функция поиска на бинаризованном изображении объектов, внешне похожих на плод томата

    image_bgr -- BGR-изображение, на которм необходимо найти плоды
    mask -- бинаризованное изображение
    blob_detector -- экземпляр Simple_Blob_Detector
    split_tomatoes -- (boolean) Включение/отключение разделения объектов на бинаризованном изображении
    min_distance_when_splitting -- минимальное расстояние между центрами искомых объектов (при разделении объектов)
    """
    if split_tomatoes:
        # С разделением соприкасающихся объектов

        # Для построения карты расстояний необходимо заполнить все внутренние отверстия объектов
        mask_copy = hole_filler.fill(mask)

        # Подробнее описано https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
        distance_map = ndimage.distance_transform_edt(mask_copy)
        local_max = peak_local_max(distance_map, indices=False, min_distance=np.uint8(min_distance_when_splitting),
                                   labels=mask_copy)
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance_map, markers, mask=mask_copy)

        keypoints = []
        for label in np.unique(labels):
            if label == 0:
                continue

            mask_with_one_object = np.zeros(mask_copy.shape, dtype="uint8")
            mask_with_one_object[labels == label] = 255

            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            image_with_one_tomato = cv2.bitwise_and(image_gray, image_gray, mask=mask_with_one_object)
            keypoint = blob_detector.detect(image_with_one_tomato)
            if len(keypoint) != 0:
                keypoints = keypoints + keypoint

        return keypoints
    else:
        # Без разделения соприкасающихся объектов
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return blob_detector.detect(cv2.bitwise_and(image_gray, image_gray, mask=mask))


def get_parameters_for_single_tomato(frame_height, frame_width):
    """
    Функция получения SimpleBlobDetector_Params по умолчанию

    Возвращает SimpleBlobDetector_Params

    frame_height -- высота исходного изображения
    frame_width -- ширина исходного изображения
    """
    parameters = cv2.SimpleBlobDetector_Params()

    parameters.filterByColor = False

    # Фильтр площади
    parameters.filterByArea = True
    parameters.minArea = frame_height / 15 * frame_width / 15
    parameters.maxArea = frame_height / 1.5 * frame_width / 1.5

    # Фильтр округлости
    parameters.filterByCircularity = True
    parameters.minCircularity = 0.6

    # Фильтр выпуклости
    parameters.filterByConvexity = True
    parameters.minConvexity = 0.5

    # Фильтр вытянутости
    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.4

    return parameters
