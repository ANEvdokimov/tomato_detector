import cv2


def detect(image, min_radius=100, max_radius=300):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=25, param2=28,
                            minRadius=min_radius, maxRadius=max_radius)


def draw_circles(image, circles):
    if circles is not None:
        for circle in circles[0, :]:
            image = cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
    return image
