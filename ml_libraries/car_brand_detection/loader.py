from helpers.logger import get_logger
from helpers.video import VideoUitls
from helpers.model import ModelLoader
from helpers.url import read_path
from ml_libraries.car_brand_detection.carlogo_df import load_carlogo_model
from ml_libraries.car_brand_detection.carlogo_df import detect_carlogo

# logger
logging = get_logger(__name__)
video_util = VideoUitls()


class CarBrandModelLoader:
    def __init__(self) -> None:
        self.counter = 0
        self.loaded_model = None
        self.drawing_response = video_util.get_default_display_frame()

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config

        logging.info('Load car_brand Model....')

        analytic_value['carlogo_model_path'] = read_path(analytic_value['carlogo_model_path'], "car_brand", "model_carlogo.pt")

        model_loaded = load_carlogo_model(
            analytic_value['carlogo_model_path'])
        new_model = ModelLoader(analytic_key, model_loaded)

        return new_model

    def extract_frame(self, model, frame,  analytics):
        frame_skip = analytics["car_brand"]["frame_skip"]
        api_responses = []
        raw_datas = []

        if self.loaded_model is None:
            self.loaded_model = model.load()

        if self.counter >= 0 and self.counter % frame_skip == 0:
            brand_logo, drawing_response, api_response, raw_data = detect_carlogo(
                self.loaded_model, frame.copy())

            self.drawing_response = video_util.get_default_display_frame()

            if brand_logo:
                # logging.debug('Car logo Extracting...')
                # logging.info('Detected car logo: %s', brand_logo)

                self.counter = 0
                self.drawing_response = drawing_response

            if api_response is not None:
                api_responses.append(api_response)

            if raw_data is not None:
                raw_datas.append(raw_data)

        self.counter += 1
        return self.drawing_response, api_responses, raw_datas
