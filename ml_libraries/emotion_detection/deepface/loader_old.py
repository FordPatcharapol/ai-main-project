from helpers.logger import get_logger
from helpers.video import VideoUitls
from helpers.model import ModelLoader
from ml_libraries.emotion_detection.emotion_df import load_emotion_model
from ml_libraries.emotion_detection.emotion_df import detect_emotion

# logger
logging = get_logger(__name__)
video_util = VideoUitls()


class EmotionModelLoader:
    def __init__(self) -> None:
        self.counter = 0
        self.emotion_model = None
        self.drawing_response = video_util.get_default_display_frame()

    def model_loader(self, analytic_key, analytic_value):
        logging.debug('Load emotion_model Model....')

        model_loaded = load_emotion_model(
            analytic_value['emotion_model_path'])
        new_model = ModelLoader(analytic_key, model_loaded)

        return new_model

    def extract_frame(self, model, frame,  analytics):
        frame_skip = analytics["emotion"]["frame_skip"]

        self.counter += 1

        if self.counter > 0 and self.counter % frame_skip == 0:
            emotion_model, drawing_response = detect_emotion(
                model.load(), frame.copy())

            self.drawing_response = video_util.get_default_display_frame()

            if emotion_model:
                logging.debug('Emotion model Extracting...')
                logging.info('Detected emotion model: %s', emotion_model)

                self.emotion_model = emotion_model
                self.counter = 0
                self.drawing_response = drawing_response

        return self.drawing_response
