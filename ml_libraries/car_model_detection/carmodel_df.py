import sys
from ultralytics import YOLO

from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.video import VideoUitls


# logger
logging = get_logger(__name__)
video_util = VideoUitls()
api_helpers = ApiHelper()
threshold = 0.5

# package parameter
this = sys.modules[__name__]
this.prev_model = ""


def load_carmodel_model(model_path):
    return YOLO(model_path)


def detect_carmodel(model, frame):
    image = frame.copy()
    car_model = []
    car_model_api = []
    coord_api = []

    results = model(image, verbose=False)
    drawing_response = video_util.get_default_display_frame()


    for result in results[0].boxes.data.tolist():
        rectangle_position = []
        text_position = []

        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            brand_model = results[0].names[int(class_id)].upper()
            car_model.append(brand_model)
            car_model_api.append({"model_name": brand_model})

            rectangle_position.append({
                "start_point": (int(x1), int(y1)),
                "end_point": (int(x2), int(y2)),
                "color": (255, 0, 0),
                "thickness": 4,
            })

            text_position.append({
                "text": brand_model,
                "coord":  (int(x1), int(y1 - 10)),
                "color": (255, 0, 0),
                "thickness": 2,
            })

            coord_api.append({
                "0": rectangle_position[0]["start_point"],
                "1": rectangle_position[0]["end_point"]
            })

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    raw_data = None
    if car_model_api:
        raw_data = api_helpers.get_payload_struct(
            "car-model",
            coord_api,
            "Detected car model: " + str(car_model),
            {"car_model": car_model_api}
        )

    api_response = None
    if this.prev_model == "" or set(this.prev_model) != set(car_model):

        logging.debug('Car model Extracting...')
        logging.info('Detected car model: %s', car_model)

        logging.info("car model changed")
        api_response = api_helpers.get_payload_struct(
            "car-model",
            coord_api,
            "Detected car model: " + str(car_model),
            {"car_model": car_model_api}
        )
        this.prev_model = car_model

    return car_model, drawing_response, api_response, raw_data
