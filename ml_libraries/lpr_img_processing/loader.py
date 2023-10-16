from helpers.model import ModelLoader
from helpers.lpr import lpr_ocr_tesseract, lpr_ocr_easy_ocr
from helpers.logger import get_logger
from helpers.video import VideoUitls

from ml_libraries.lpr_img_processing.LPR_df import detect_img

# logger
logging = get_logger(__name__)
video_util = VideoUitls()


class LprOcrModelLoader:
    def __init__(self) -> None:
        self.counter = 0
        self.plate = None
        self.country = None
        self.drawing_response = video_util.get_default_display_frame()

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config
        _ = analytic_key
        _ = analytic_value
        logging.info('Load lpr_ocr Model....')
        new_model = ModelLoader(analytic_key, None)

        return new_model

    def extract_frame(self, model, frame, analytics):
        _ = model
        _ = analytics

        image = frame.copy()

        ocr_selection = analytics["lpr_ocr"]["ocr_model"]
        frame_skip = analytics["lpr_ocr"]["frame_skip"]

        self.counter += 1

        if self.counter > 0 and self.counter % frame_skip == 0:
            self.counter = 0
            lpr_image, is_detected, drawing_response = detect_img(frame, image)

            if is_detected:
                self.drawing_response = drawing_response
                if ocr_selection == "tesseract":
                    plate, country = lpr_ocr_tesseract(lpr_image["frame"])
                elif ocr_selection == "jaided":
                    plate, country = lpr_ocr_easy_ocr(lpr_image["frame"])
                else:
                    logging.error("Incorrect OCR model setting for lpr_ocr")
                    return None

                if plate is not None:
                    logging.info(' > plate: %s , Country: %s',
                                 plate, country)
                    self.plate = plate
                    self.country = country

                self.counter = 0
            else:
                self.drawing_response = video_util.get_default_display_frame()

        return self.drawing_response
