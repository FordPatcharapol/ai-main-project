import cv2
import joblib
import numpy as np
import sys
from tensorflow.keras.models import load_model

from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.video import VideoUitls

from ml_libraries.emotion_detection.model.utils import align_face, bb_to_rect, preprocess_img
from ultralytics import YOLO

# logger
logging = get_logger(__name__)

video_util = VideoUitls()
api_helpers = ApiHelper()

bgr_green = (0, 255, 0)
bgr_red = (0, 0, 255)

desiredLeftEye=(0.32, 0.32)

conf_threshold = 0.7

# package parameter
this = sys.modules[__name__]
this.prev_emotoin = ""


def load_face_model(model_path):
    return YOLO(model_path)

def load_emotion_model(model_path):
    model = load_model(model_path)
    return model

def load_label2text(label2text_path):
    label2text = joblib.load(label2text_path)
    return label2text

def detect_emotion(frame,face_model, emotion_model, label2text):
    image = frame.copy()
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotion = []
    bboxes = []
    idx = 0
    offset = 30
    x_pos, y_pos = 20, 140

    old_width, old_height = 1920, 1080
    new_width, new_height = frame.shape[1], frame.shape[0]

    x_ratio = new_width / old_width
    y_ratio = new_height / old_height

    emotion_api = []
    coord_api = []

    results = face_model.predict(image, conf=0.7, verbose=False)
    drawing_response = video_util.get_default_display_frame()

    for face in results[0].boxes.data.tolist():
        xmin, ymin, xmax, ymax = face[0], face[1], face[2], face[3]

        rectangle_position = []
        text_position = []
        idx += 1
        bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        face = [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]

        img_arr = align_face(gray_frame, bb_to_rect(face), desiredLeftEye)
        img_arr = preprocess_img(img_arr, resize=False)

        predicted_proba = emotion_model.predict(img_arr,verbose=None)
        predicted_label = np.argmax(predicted_proba[0])

        rectangle_position.append({
            "start_point": (int(xmin), int(ymin)),
            "end_point": (int(xmax), int(ymax)),
            "color": bgr_green,
            "thickness": 2,
        })

        coord_api.append({
            "0": rectangle_position[0]["start_point"],
            "1": rectangle_position[0]["end_point"]
        })

        emotion_text = f"Person {idx}: {label2text[predicted_label]}"
        emotion.append(emotion_text)

        text_position.append({
            "text": emotion_text,
            "coord": (int(xmin + 5), int(ymin-10)),
            "color": bgr_green,
            "thickness": 2,
        })

        emotion_api.append({"emotion": str(label2text[predicted_label]), "person": str(idx)})

        detail_text = f"Person {idx} :  "
        y_pos = y_pos + 2*offset

        text_position.append({
            "text": detail_text,
            "coord": (int(x_pos*x_ratio), int(y_pos*y_ratio)),
            "color": bgr_green,
            "thickness": 2,
        })

        for k,v in label2text.items():
            perc_text = f"{v}: {round(predicted_proba[0][k]*100, 3)}%"
            y_pos = y_pos + offset

            text_position.append({
                "text": perc_text,
                "coord": (int(x_pos*x_ratio), int(y_pos*y_ratio)),
                "color": bgr_green,
                "thickness": 2,
            })

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    raw_data = None
    if emotion_api:
        raw_data = api_helpers.get_payload_struct(
                "mood-tone",
                coord_api,
                "Detected emotion: " + str(emotion),
                {"cur_emotion": emotion_api}
            )

    api_response = None
    if this.prev_emotoin == "" or set(this.prev_emotoin) != set(emotion):
        if emotion != []:
            logging.debug('Emotion model Extracting...')
            logging.info('Detected emotion model: %s', emotion)

            logging.info("emotion changed")
            api_response = api_helpers.get_payload_struct(
                "mood-tone",
                coord_api,
                "Detected emotion: " + str(emotion),
                {"cur_emotion": emotion_api}
            )
            this.prev_emotoin = emotion

    return emotion, drawing_response, api_response, raw_data
