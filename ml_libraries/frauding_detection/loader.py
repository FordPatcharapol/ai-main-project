import sys
import time

from ml_libraries.frauding_detection.pose import load_pose_model
from ml_libraries.frauding_detection.detect import load_detect_model, extract_fraud

from helpers.api import ApiHelper
from helpers.lpr import lpr_ocr_tesseract, lpr_ocr_easy_ocr
from helpers.model import ModelLoader
from helpers.logger import get_logger
from helpers.video import VideoUitls
from helpers.url import read_path

# logger
logging = get_logger(__name__)
video_util = VideoUitls()
api_helpers = ApiHelper()

# package parameter
this = sys.modules[__name__]
this.prev_status = ""
this.founding_time = []
this.this.before_time = time.time()
this.prev_item = []


class FraudModelLoader:
    def __init__(self) -> None:
        self.loaded_model = None
        self.counter = 0
        self.drawing_response = video_util.get_default_display_frame()
        self.api_response = []

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config
        logging.info('Load frauding Model...')
        _ = analytic_key

        main_model = ModelLoader(analytic_key, [])

        analytic_value['detect_model_path'] = read_path(
            analytic_value['detect_model_path'], "frauding", "frauding.pt")
        model_loaded = load_detect_model(
            model_path=analytic_value['detect_model_path'])
        new_model = ModelLoader(analytic_key + "_detect", model_loaded)
        main_model.model.append(new_model)

        analytic_value['pose_model_path'] = read_path(
            analytic_value['pose_model_path'], "yolov8s-pose", "yolov8s-pose.pt")
        model_loaded = load_pose_model(
            model_path=analytic_value['pose_model_path'])
        new_model = ModelLoader(analytic_key + "_pose", model_loaded)
        main_model.model.append(new_model)

        return main_model

    def extract_frame(self, frame, model, analytics):
        frame_skip = analytics["frauding"]["frame_skip"]
        time_range = analytics["frauding"]["time_range"]

        if self.loaded_model is None:
            self.loaded_model = model.load()
        model_lst = model.model

        detect_model_idx = [index for index, obj in enumerate(
            model_lst) if obj.name == 'frauding_detect']
        pose_model_idx = [index for index, obj in enumerate(
            model_lst) if obj.name == 'frauding_pose']

        detect_model = model_lst[int(detect_model_idx[0])]
        pose_model = model_lst[int(pose_model_idx[0])]

        image = frame.copy()
        self.counter += 1
        this.founding_time = []
        coord = []
        text = ""

        raw_datas = []

        if self.counter > 0 and self.counter % frame_skip == 0:
            self.counter = 0
            self.api_response = []

            person_lst, self.drawing_response = extract_fraud(
                image, detect_model.model, pose_model.model)

            if person_lst is not None:
                for person in person_lst:
                    if person.is_thief and person.timestamp - this.before_time > time_range:
                        this.before_time = person.timestamp

                    this.prev_item.append(person.is_thief)

                    this.founding_time.append({
                        'timestamp': int(person.timestamp),
                        'type': 'steal',
                        'level': 'warning'
                    })

                    coord.append({
                        "0": [person.bbox[0], person.bbox[1]],
                        "1": [person.bbox[2], person.bbox[3]]
                    })

                    self.counter = 0

            else:
                self.drawing_response = video_util.get_default_display_frame()

        text += 'Frauded: ' + str(this.founding_time)

        if this.prev_status == "" or this.prev_status != str(this.prev_item) and any(this.prev_item):
            logging.info("Frauding Detected")
            self.api_response.append(
                api_helpers.get_payload_struct(
                    "frauding",
                    coord,
                    str(text),
                    {"frauding": this.founding_time}
                )
            )

            this.prev_status = str(this.prev_item)
            this.prev_item = []
            this.founding_time = []
        else:
            self.api_response = []

        return self.drawing_response, self.api_response, raw_datas
