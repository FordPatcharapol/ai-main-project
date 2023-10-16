import sys

from ml_libraries.multi_cam_tracking.tracker import load_detect_model, find_object, update_tracker, object_searching

from helpers.model import ModelLoader
from helpers.video import VideoUitls
from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.url import read_path

# logger
logging = get_logger(__name__)
api_helpers = ApiHelper()

# package parameter
this = sys.modules[__name__]

video_util = VideoUitls()


class MultiCameraModelLoader:
    def __init__(self) -> None:
        self.loaded_model = None
        self.counter = 0
        self.api_response = []
        self.drawing_response = video_util.get_default_display_frame()

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config
        analytic_value['detect_model_path'] = read_path(analytic_value['detect_model_path'], "multi_cam", "yolov8s.pt")

        model_loaded = load_detect_model(
            model_path=analytic_value['detect_model_path'])
        new_model = ModelLoader(analytic_key, model_loaded)

        return new_model

    def extract_frame(self, frame, model, analytics):
        frame_skip = analytics["multi_cam"]["frame_skip"]
        classes = analytics["multi_cam"]["class"]
        self.counter += 1
        self.api_response = []
        raw_datas = []


        if self.counter > 0 and self.counter % frame_skip == 0:
            image = frame.copy()

            self.drawing_response = video_util.get_default_display_frame()

            if self.loaded_model is None:
                self.loaded_model = model.load()

            object_lst = find_object(image, self.loaded_model, classes)

            if len(object_lst) == 0:
                return self.drawing_response, self.api_response, raw_datas

            tracker_lst = update_tracker(object_lst)

            self.drawing_response, self.api_response = object_searching(tracker_lst, image)

        return self.drawing_response, self.api_response, raw_datas
