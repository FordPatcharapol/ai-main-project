from helpers.lpr import lpr_ocr_easy_ocr
from helpers.video import VideoUitls
from ultralytics import YOLO

video_util = VideoUitls()
bgr_green = (0, 255, 0)


# Model
def load_lpr_model(model_path):
    model = YOLO(model_path)
    return model


def extract_lpr(frame, model):
    lprs = []
    cropped_image = None

    drawing_response = video_util.get_default_display_frame()

    # Inference
    results = model(frame, verbose=False)
    objects = results[0].boxes.data.tolist()

    for row in objects:
        rectangle_position = []
        text_position = []

        xmin, ymin, xmax, ymax = int(row[0]), int(
            row[1]), int(row[2]), int(row[3])
        cropped_image = frame[ymin:ymax, xmin:xmax]
        lprs.append({
            "frame": cropped_image,
            "coord": (xmin, ymin, xmax, ymax)
        })

        rectangle_position.append({
            "start_point": (int(xmin), int(ymin)),
            "end_point": (int(xmax), int(ymax)),
            "color": bgr_green,
            "thickness": 3,
        })

        text_position.append({
            "text": 'license plate',
            "coord":  (int(xmin), int(ymin - 10)),
            "color": bgr_green,
            "thickness": 2,
        })

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    # print('drawing_response', drawing_response)
    return lprs, drawing_response
