import sys

from ml_libraries.lpr_deep_learning.detect import load_lpr_model, extract_lpr

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
this.prev_lpr = ""
this.aval_lpr = []


class LprDlModelLoader:
    def __init__(self) -> None:
        self.loaded_model = None
        self.counter = 0
        self.frame_skip = None
        self.drawing_response = video_util.get_default_display_frame()
        self.plate = None
        self.country = None
        self.api_response = []

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config
        logging.info('Load lpr_dl Model...')
        _ = analytic_key

        analytic_value['detect_model_path'] = read_path(
            analytic_value['detect_model_path'], "lpr_dl", "detect_LP.pt")

        model_loaded = load_lpr_model(
            model_path=analytic_value['detect_model_path'])
        new_model = ModelLoader(analytic_key, model_loaded)

        return new_model

    def extract_frame(self, frame, model, analytics):
        frame_skip = analytics["lpr_dl"]["frame_skip"]
        ocr_selection = analytics["lpr_dl"]["ocr_model"]

        self.frame_skip = frame_skip

        if self.loaded_model is None:
            self.loaded_model = model.load()

        image = frame.copy()

        self.counter += 1

        coord = []
        text = ""

        raw_datas = []

        if self.counter >= 0 and self.counter % frame_skip == 0:
            self.counter = 0
            self.api_response = []
            this.aval_lpr = []

            lpr_image, self.drawing_response = extract_lpr(
                image, self.loaded_model)

            if lpr_image is not None:

                for lpr_i in lpr_image:
                    if ocr_selection == "tesseract":
                        plate, country = lpr_ocr_tesseract(lpr_i['frame'])
                    elif ocr_selection == "jaided":
                        plate, country = lpr_ocr_easy_ocr(lpr_i['frame'])
                    else:
                        logging.error(
                            "Incorrect OCR model setting for lpr_ocr")
                        return video_util.get_default_display_frame()

                    if plate is not None:
                        logging.info(' > plate: %s , Country: %s',
                                     plate[0], country)
                        self.drawing_response["text"] += [{
                            "text": "[" + plate[0]+" : " + country + "]",
                            "coord":  (int(lpr_i['coord'][0]+80), int(lpr_i['coord'][1] - 10)),
                            "color": (0, 255, 0),
                            "thickness": 2,
                        }]

                        this.aval_lpr.append({
                            "plate": plate[0],
                            "province": country
                        })

                        coord.append({
                            "0": [lpr_i['coord'][0], lpr_i['coord'][1]],
                            "1": [lpr_i['coord'][2], lpr_i['coord'][3]]
                        })

                        text += 'Plate: ' + plate[0] + \
                            ', Country: ' + country + ' '

                        self.plate = plate
                        self.country = country
                        self.counter = 0

            else:
                self.drawing_response = video_util.get_default_display_frame()

        if this.aval_lpr and coord:
            raw_datas.append(
                api_helpers.get_payload_struct(
                    "lpr-dl",
                    coord,
                    "LPR: [" + str(text) + "]",
                    {"license_plates": this.aval_lpr}
                )
            )

        if this.prev_lpr == "" or this.prev_lpr != str(this.aval_lpr):
            logging.info("License Plate Changed")
            self.api_response.append(
                api_helpers.get_payload_struct(
                    "lpr-dl",
                    coord,
                    "LPR: [" + str(text) + "]",
                    {"license_plates": this.aval_lpr}
                )
            )

            this.prev_lpr = str(this.aval_lpr)

        else:
            self.api_response = []

        return self.drawing_response, self.api_response, raw_datas
