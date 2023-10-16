import time

from ultralytics import YOLO
from helpers.video import VideoUitls
from ml_libraries.frauding_detection.pose import is_thief, ObjectInfo

video_util = VideoUitls()

def load_detect_model(model_path):
    model = YOLO(model_path)
    return model


def extract_fraud(frame, detect_model, pose_model):

    # Inference
    shoplifters = detect_model(source=frame, conf=0.5, verbose=False)
    shoplifters_bbox = shoplifters[0].boxes.data.tolist()

    person_lst = []
    founding_time = []
    drawing_response = video_util.get_default_display_frame()

    for shoplifter in shoplifters_bbox:
        rectangle_position = []
        text_position = []
        bgr_green = (0, 255, 0)
        msg = 'Not Frauding'

        xmin, ymin, xmax, ymax = int(shoplifter[0]), int(
            shoplifter[1]), int(shoplifter[2]), int(shoplifter[3]),
        crop_img = frame[ymin:ymax, xmin:xmax]

        results = pose_model(source=crop_img, conf=0.5, verbose=False)

        result_keypoint = results[0].keypoints.xy.tolist()[0]

        if len(results[0].keypoints.xy.tolist()[0]) < 5:
            continue

        fraud_status = is_thief(result_keypoint)

        new_object_info = ObjectInfo(
            [xmin, ymin, xmax, ymax], fraud_status, time.time())
        person_lst.append(new_object_info)

        if fraud_status:
            bgr_green = (0, 0, 255)
            msg = 'Frauding'

        rectangle_position.append({
            "start_point": (int(xmin), int(ymin)),
            "end_point": (int(xmax), int(ymax)),
            "color": bgr_green,
            "thickness": 3,
        })

        text_position.append({
            "text": msg,
            "coord":  (int(xmin), int(ymin - 10)),
            "color": bgr_green,
            "thickness": 2,
        })

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    return person_lst, drawing_response
