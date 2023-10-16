import sys
from helpers.model import ModelLoader
from helpers.logger import get_logger
from helpers.video import VideoUitls
from helpers.url import read_path


from ml_libraries.parking_space.parking_df import load_parking_model
from ml_libraries.parking_space.parking_df import parking_space, class_coco, get_rois, resize_rois

# package parameter
this = sys.modules[__name__]
this.prev_slots = ""

# logger
logging = get_logger(__name__)

video_util = VideoUitls()


class ParkingSpaceModelLoader:
    def __init__(self) -> None:
        self.counter = 0
        self.loaded_model = None
        self.drawing_response = video_util.get_default_display_frame()
        self.rois = None
        self.list_slot = []
        self.class_list = None

    def model_loader(self, source_config, analytic_key, analytic_value):
        logging.info('Load parking_space Model....')

        analytic_value['parking_model_path'] = read_path(
            analytic_value['parking_model_path'], "parking_space", "yolov8s.pt")
        model_loaded = load_parking_model(
            analytic_value['parking_model_path'])
        new_model = ModelLoader(analytic_key, model_loaded)

        analytic_value['class_path'] = read_path(
            analytic_value['class_path'], "parking_space", "coco.txt")
        analytic_value['rois_path'] = read_path(
            analytic_value['rois_path'], "parking_space", "rois.csv")

        self.open_rois(analytic_value, source_config['resize_frame'])
        self.class_list = class_coco(analytic_value['class_path'])

        return new_model

    def extract_frame(self, model, frame, analytics):
        frame_skip = analytics["parking_space"]["frame_skip"]
        class_detect = analytics["parking_space"]["class_detect"]

        api_responses = []
        raw_datas = []

        if self.loaded_model is None:
            self.loaded_model = model.load()

        if self.counter >= 0 and self.counter % frame_skip == 0:
            num_aval, aval_spot, drawing_response, api_response, raw_data = parking_space(
                self.loaded_model, frame, self.class_list, self.rois, class_detect)

            if this.prev_slots != aval_spot:
                logging.debug('Parkingspace Extracting...')
                logging.info(
                    'Number of Available Spots: %s, Available spots: %s', num_aval,  aval_spot)
                this.prev_slots = aval_spot

            self.counter = 0
            self.drawing_response = drawing_response

            if api_response is not None:
                api_responses.append(api_response)

            # if raw_data is not None:
            #     raw_datas.append(raw_data)

        self.counter += 1
        return self.drawing_response, api_responses, raw_datas

    def open_rois(self, analytic_value, new_size):
        if self.rois is not None:
            return None

        original_rois = get_rois(analytic_value['rois_path'])
        self.rois = resize_rois(original_rois, new_size)

        return None
