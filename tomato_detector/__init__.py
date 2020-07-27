import cv2
from tomato_detector import image_processor
from tomato_detector import color_filter
from tomato_detector import form_filter

# Цветовой диапазон по умолчанию
DEFAULT_COLOR_SETTINGS = {"red": {"min": (0, 132, 0), "max": (179, 255, 255)}}


def detect_in_image(image, color_ranges_hsv=None, equalize_light=True, color_balance=True, split_tomatoes=True,
                    simple_blob_detector_params=None, min_distance_when_splitting=0):
    """
    Функция обнаружения плодов томата на изображении

    Возвращает расположения плодов и бинаризованное изображение

    Пример словаря с пользовательскими цветовыми диапазонами:
    color_ranges = {"red": {"min": (0, 132, 0), "max": (179, 255, 255)},
                    "green": {"min": (32, 50, 40), "max": (50, 255, 255)}}

    image -- исходное BGR-изображение
    color_ranges_hsv -- (необязательное) именовынные цветовые диапазоны для поиска
    equalize_light -- (boolean, необязательное) включение/отключение выравнивания контраста
    color_balance -- (boolean, необязательное) включение/отключение корректировки баланса белого
    split_tomatoes -- (boolean, необязательное) включение/отключение разделения соприкасающихся плодов
    simple_blob_detector_params -- (необязательное) пользовательские параметры Simple_Blob_Detector (OpenCV)
    """
    if color_ranges_hsv is None:
        color_ranges_hsv = DEFAULT_COLOR_SETTINGS

    if equalize_light:
        image = image_processor.equalize_light(image)
    if color_balance:
        image = image_processor.white_balance(image)

    if simple_blob_detector_params is None:
        simple_blob_detector_params = form_filter.get_parameters_for_single_tomato(image.shape[0], image.shape[1])
    blob_detector = cv2.SimpleBlobDetector_create(simple_blob_detector_params)

    # удаление шума
    image_source_blur = cv2.bilateralFilter(image, 5, 75, 75)

    global mask_tomatoes_processed
    tomato_keypoints = {}
    for name, color_range in color_ranges_hsv.items():
        # Поиск по цвету
        mask_tomatoes = color_filter.find_tomatoes_by_color(image_source_blur, color_range)
        # Удаление шумов с бинаризованного изображения
        mask_tomatoes_processed = cv2.morphologyEx(mask_tomatoes, cv2.MORPH_OPEN, (50, 50), iterations=5)
        # Разделение объектов и фильрация по внешней форме
        tomato_keypoints[name] = \
            form_filter.find_tomatoes(image_source_blur, mask_tomatoes_processed, blob_detector, split_tomatoes,
                                      min_distance_when_splitting)

    return tomato_keypoints, mask_tomatoes_processed
