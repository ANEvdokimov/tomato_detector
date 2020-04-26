import cv2
from Main import blob_detector
# from Main import circles_detector
from Main import iem
from Main import hole_filler


vid = cv2.VideoCapture("../Dataset/1_720.mp4")
# frame_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

while vid.isOpened():
    _, image_source = vid.read()
    if image_source is None:
        break

    image_source_equalized_light = iem.equalize_light(image_source)
    image_source_blur = cv2.GaussianBlur(image_source_equalized_light, (5, 5), 0)
    # image_source_blur = cv2.medianBlur(image_source, 5)
    # cv2.imshow("wb", image_source_blur)

    image_bgr = image_source_blur
    image_hsv = cv2.cvtColor(image_source_blur, cv2.COLOR_BGR2HSV)
    image_Lab = cv2.cvtColor(image_source_blur, cv2.COLOR_BGR2Lab)
    # image_Luv = cv2.cvtColor(image_source_blur, cv2.COLOR_BGR2Luv)
    image_YCrCb = cv2.cvtColor(image_source_blur, cv2.COLOR_BGR2YCrCb)

    # first condition: b*<138 => background
    mask_background = cv2.inRange(image_Lab, (0, 1, 1), (255, 255, 138))
    mask_without_background = cv2.bitwise_not(mask_background)

    # second condition: r>=72 => without leaf
    image_bgr_without_background = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_without_background)
    mask_without_background_leafs = cv2.inRange(image_bgr_without_background, (0, 0, 72), (255, 255, 255))
    # cv2.imshow("without_background_leafs",
    #             cv2.bitwise_and(image_source, image_source, mask=mask_without_background_leafs))

    # third condition: S>=132 => fruit
    image_hsv_without_background_and_leafs = cv2.bitwise_and(image_hsv, image_hsv, mask=mask_without_background_leafs)
    mask_red_fruit = cv2.inRange(image_hsv_without_background_and_leafs, (0, 132, 0), (179, 255, 255))
    mask_without_background_leafs_fruit1 = cv2.subtract(mask_without_background_leafs, mask_red_fruit)
    # cv2.imshow("without_fruit1",
    #             cv2.bitwise_and(image_source, image_source, mask=mask_without_background_leafs_fruit1))

    # fourth condition: S<106
    image_hsv_without_background_leafs_fruit1 = \
        cv2.bitwise_and(image_hsv, image_hsv, mask=mask_without_background_leafs_fruit1)
    mask_without_background_leafs_fruit1_lessS106 = \
        cv2.inRange(image_hsv_without_background_leafs_fruit1, (0, 0, 1), (179, 105, 255))
    mask_without_background_leafs_fruit1_moreS106 = \
        cv2.inRange(image_hsv_without_background_leafs_fruit1, (0, 106, 0), (179, 255, 255))
    # cv2.imshow("less106",
    #             cv2.bitwise_and(image_source, image_source, mask=mask_without_background_leafs_fruit1_lessS106))
    # cv2.imshow("more106",
    #             cv2.bitwise_and(image_source, image_source, mask=mask_without_background_leafs_fruit1_moreS106))

    # fifth condition: R>=104 and S<106 => fruit
    image_bgr_without_background_leafs_fruit1_lessS106 = \
        cv2.bitwise_and(image_bgr, image_bgr, mask=mask_without_background_leafs_fruit1_lessS106)
    mask_fruit2 = cv2.inRange(image_bgr_without_background_leafs_fruit1_lessS106, (0, 0, 104), (255, 255, 255))
    # cv2.imshow("fruit2", cv2.bitwise_and(image_source, image_source, mask=mask_fruit2))

    # sixth and seventh conditions: Cr>=128 and Cb<102 and S>106 => fruit
    image_YCrCb_background_leafs_fruit1_moreS106 = \
        cv2.bitwise_and(image_YCrCb, image_YCrCb, mask=mask_without_background_leafs_fruit1_moreS106)
    mask_fruit3 = cv2.inRange(image_YCrCb_background_leafs_fruit1_moreS106, (0, 128, 0), (255, 255, 101))
    # cv2.imshow("fruit3", cv2.bitwise_and(image_source, image_source, mask=mask_fruit3))

    mask_fruit = cv2.add(mask_red_fruit, mask_fruit2)
    mask_fruit = cv2.add(mask_fruit, mask_fruit3)
    # mask_fruit = cv2.add(mask_red_fruit, mask_fruit3)
    # cv2.imshow("all_fruit", cv2.bitwise_and(image_source, image_source, mask=mask_fruit))

    mask_fruit_erode = cv2.erode(mask_red_fruit, (30, 30), iterations=5)    # TODO is used only red
    # mask_fruit_dilate = cv2.dilate(mask_fruit_erode, (5, 5), iterations=2)
    mask_fruit_dilate = mask_fruit_erode
    # cv2.imshow('fruit', cv2.bitwise_and(image_source, image_source, mask=mask_fruit_dilate))

    # Using circles detector
    # circles = circles_detector.detect(
    #     cv2.bitwise_and(image_source, image_source, mask=mask_fruit_dilate), max_radius=int(frame_height/3))
    # image_with_circles = \
    #     circles_detector.draw_circles(cv2.bitwise_and(image_source, image_source, mask=mask_fruit_dilate), circles)

    # Using simple blob detector
    # mask_fruit_dilate_without_holes = hole_filler.fill(mask_fruit_dilate)
    # blobs = blob_detector.detect(mask_fruit_dilate)
    # image_with_circles = blob_detector.draw_keypoints(mask_fruit_dilate, blobs)
    blobs = \
        blob_detector.detect(cv2.bitwise_and(image_source, image_source, mask=mask_fruit_dilate))
    image_with_circles = blob_detector.draw_keypoints(image_source, blobs)
    # cv2.imshow("circle", image_with_circles)

    cv2.imshow('detected circles', image_with_circles)
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
