import cv2
import numpy as np
import imutils
import easyocr

from helpers.video import VideoUitls

# init OCR
reader = easyocr.Reader(['th'])

# init pytesseract
scale = 0.5

bgr_green = (0, 255, 0)
bgr_pink = (127, 0, 255)
white = 255
black = 0
plate_th = 80
plate_size = (400, 175)

video_util = VideoUitls()


def get_contour_angle(contour):
    # Find minimum area rectangle
    rect = cv2.minAreaRect(contour)

    # Calculate the angle of the rectangle
    angle = rect[-1]

    # Convert the angle to the range of [0, 180)
    if angle < -45:
        angle += 90

    return angle


def detect_img(frame, frame_c):
    drawing_response = video_util.get_default_display_frame()

    # preprocessing

    display = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
    display_gray = cv2.bilateralFilter(display, -1, 15, 5)

    edged = cv2.Canny(display_gray, 122, 186)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screen_contour = None

    mask = np.zeros(display_gray.shape, np.uint8)

    # start detect LP
    for c in cnts:
        # approximate the contour
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:

            # if peri > 80 and area > 4000:
            if peri > 100 and area > 5000:
                screen_contour = approx
                cv2.drawContours(mask, [screen_contour], black, white, -1)
                contour = c

        else:
            continue

    cv2.bitwise_and(display, display, mask=mask)
    (y, x) = np.where(mask == white)  # <---

    x = x.tolist()
    y = y.tolist()

    rectangle_position = []
    text_position = []
    resp = {}

    if len(x) != 0:
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        width_box = bottomx - topx
        height_box = bottomy - topy
        ratio = float(width_box/height_box)  # <---

        if ratio >= 1.5 and ratio <= 3:
            rectangle_position.append({
                "start_point": (topx, topy),
                "end_point":  (bottomx+1, bottomy+1),
                "color": bgr_green,
                "thickness": 3,
            })
            text_position.append({
                "text": 'license plate',
                "coord":   (topx + 20, topy - 20),
                "color": bgr_pink,
                "thickness": 2,
            })

            drawing_response["rectangle"] += rectangle_position
            drawing_response["text"] += text_position

            cropped_img = display[topy:bottomy+1, topx:bottomx+1]

            # alignment LP
            cropped_img = align_lp(contour, cropped_img, display, x, y)

            _, thresh = cv2.threshold(
                cropped_img, plate_th, white, cv2.THRESH_BINARY)

            # make thicker character
            kernel = np.ones(5, np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            resp["frame"] = thresh
            resp["coord"] = (topx, topy, bottomx, bottomy)
            # cv2.imshow("Cropped", thresh)

            return thresh, True, drawing_response
    return {"frame": frame}, False, drawing_response


def align_lp(contour, cropped_img, display, x, y):

    angle = get_contour_angle(contour)

    if angle >= 3 and angle <= 87:
        if angle <= 45:
            x1, y1 = x[np.argmin(y)], np.min(y)
            x2, y2 = np.min(x), y[np.argmin(x)]
            x3, y3 = x[np.argmax(y)], np.max(y)
            x4, y4 = np.max(x), y[np.argmax(x)]

        else:
            x1, y1 = np.min(x), y[np.argmin(x)]
            x2, y2 = x[np.argmax(y)], np.max(y)
            x3, y3 = np.max(x), y[np.argmax(x)]
            x4, y4 = x[np.argmin(y)], np.min(y)

        xy1 = np.float32(
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # <---
        xy2 = np.float32([[0, 0], [0, plate_size[1]], [
                         plate_size[0], plate_size[1]], [plate_size[0], 0]])

        mat = cv2.getPerspectiveTransform(xy1, xy2)
        cropped_img = cv2.warpPerspective(display, mat, plate_size)

    else:
        cropped_img = cv2.resize(cropped_img, plate_size)

    return cropped_img
