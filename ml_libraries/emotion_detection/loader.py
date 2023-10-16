from helpers.logger import get_logger
from helpers.video import VideoUitls
from helpers.model import ModelLoader
from helpers.url import read_path
from ml_libraries.emotion_detection.emotion_df import load_face_model, load_emotion_model, load_label2text
from ml_libraries.emotion_detection.emotion_df import detect_emotion

# logger
logging = get_logger(__name__)
video_util = VideoUitls()


class EmotionModelLoader:
    def __init__(self) -> None:
        self.counter = 0
        self.loaded_model = None
        self.drawing_response = video_util.get_default_display_frame()

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config
        logging.info('Load emotion_model Model....')

        # emotion main model
        main_model = ModelLoader(analytic_key, [])

        # face model
        analytic_value['face_model_path'] = read_path(analytic_value['face_model_path'], "emotion", "face_detect.pt")
        model_loaded = load_face_model(analytic_value['face_model_path'])
        new_model = ModelLoader(analytic_key + '_face', model_loaded)
        main_model.model.append(new_model)

        # emotion model
        analytic_value['emotion_model_path'] = read_path(analytic_value['emotion_model_path'], "emotion", "CNNModel_feraligned+ck_5emo.h5")
        model_loaded = load_emotion_model(analytic_value['emotion_model_path'])
        new_model = ModelLoader(analytic_key + '_emotion', model_loaded)
        main_model.model.append(new_model)

        # label2text file
        analytic_value['label2text_path'] = read_path(analytic_value['label2text_path'], "emotion", "label2text_CNNModel_feraligned+ck_5emo.pkl")
        model_loaded = load_label2text(analytic_value['label2text_path'])
        new_model = ModelLoader(analytic_key + '_label2text', model_loaded)
        main_model.model.append(new_model)

        return main_model

    def extract_frame(self, model, frame,  analytics):
        frame_skip = analytics["emotion"]["frame_skip"]
        api_responses = []
        raw_datas = []

        if self.loaded_model is None:
            self.loaded_model = model.load()

        if self.counter >= 0 and self.counter % frame_skip == 0:

            model_lst = model.model
            face_model_idx = [index for index, obj in enumerate(
                model_lst) if obj.name == 'emotion_face']
            emotion_model_idx = [index for index, obj in enumerate(
                model_lst) if obj.name == 'emotion_emotion']
            label2text_idx = [index for index, obj in enumerate(
                model_lst) if obj.name == 'emotion_label2text']

            face_model = model_lst[int(face_model_idx[0])]
            emotion_model = model_lst[int(emotion_model_idx[0])]
            label2text = model_lst[int(label2text_idx[0])]

            emotion_model, drawing_response, api_response, raw_data = detect_emotion(
                frame, face_model.load(), emotion_model.load(), label2text.load())

            self.drawing_response = video_util.get_default_display_frame()

            if emotion_model:
                # logging.debug('Emotion model Extracting...')
                # logging.info('Detected emotion model: %s', emotion_model)

                self.counter = 0
                self.drawing_response = drawing_response

            if api_response is not None:
                api_responses.append(api_response)

            if raw_data is not None:
                raw_datas.append(raw_data)

        self.counter += 1
        return self.drawing_response, api_responses, raw_datas
