import cv2


def find_tomatoes_by_color(image_bgr, color_range_hsv):
    """
    Функция цветового поиска

    Возвращает бинаризованное изображение

    image_bgr -- BGR-изображение
    color_range_hsv -- цветовой диапазон
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_Lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    # Удаление фона
    mask_background = cv2.inRange(image_Lab, (0, 1, 1), (255, 255, 138))
    mask_without_background = cv2.bitwise_not(mask_background)

    # Удаление листьев
    image_bgr_without_background = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_without_background)
    mask_without_background_leafs = cv2.inRange(image_bgr_without_background, (0, 0, 72), (255, 255, 255))

    # Поиск по цвету
    image_hsv_without_background_and_leafs = cv2.bitwise_and(image_hsv, image_hsv, mask=mask_without_background_leafs)
    mask_red_tomatoes = cv2.inRange(
        image_hsv_without_background_and_leafs,
        color_range_hsv["min"],
        color_range_hsv["max"]
    )

    return mask_red_tomatoes
