import sys
from ml_libraries.gender_detection.model import load_face_model, load_gender_model, load_age_model
from ml_libraries.gender_detection.gender import find_age, find_gender, find_faces, personFace

from helpers.api import ApiHelper
from helpers.video import VideoUitls
from helpers.model import ModelLoader
from helpers.logger import get_logger
from helpers.url import read_path

# logger
logging = get_logger(__name__)
video_util = VideoUitls()
bgr_green = (0, 255, 0)
api_helpers = ApiHelper()

# package parameter
this = sys.modules[__name__]
this.age_gender = ""
this.aval_age_gender = []

class GenderModelLoader:
    def __init__(self) -> None:
        self.counter = 0
        self.api_response = []
        self.drawing_response = video_util.get_default_display_frame()
        self.face_model = None
        self.age_model = None
        self.gender_model = None

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config

        logging.debug('Load Gender Model....')

        # gender main model
        main_model = ModelLoader(analytic_key, [])

        # face model
        analytic_value['face_model_path'] = read_path(analytic_value['face_model_path'], "gender", "face_detect.pt")
        model_loaded = load_face_model(analytic_value['face_model_path'])
        new_model = ModelLoader(analytic_key + '_face', model_loaded)
        main_model.model.append(new_model)

        # gender model
        analytic_value['gender_model_path']['prototxt'] = read_path(analytic_value['gender_model_path']['prototxt'], "gender", "gender.prototxt")
        analytic_value['gender_model_path']['caffe'] = read_path(analytic_value['gender_model_path']['caffe'], "gender", "gender.caffemodel")

        model_loaded = load_gender_model(
            analytic_value['gender_model_path']['prototxt'], analytic_value['gender_model_path']['caffe'])
        new_model = ModelLoader(analytic_key + '_gender', model_loaded)
        main_model.model.append(new_model)

        # age model
        analytic_value['age_model_path']['prototxt'] = read_path(analytic_value['age_model_path']['prototxt'], "gender", "age.prototxt")
        analytic_value['age_model_path']['caffe'] = read_path(analytic_value['age_model_path']['caffe'], "gender", "age.caffemodel")

        model_loaded = load_age_model(
            analytic_value['age_model_path']['prototxt'], analytic_value['age_model_path']['caffe'])
        new_model = ModelLoader(analytic_key + '_age', model_loaded)
        main_model.model.append(new_model)

        return main_model

    def extract_frame(self, frame, model, analytics):
        frame_skip = analytics["gender"]["frame_skip"]
        self.counter += 1
        this.aval_age_gender = []
        coord = []
        text = ""
        raw_datas = []

        raw_datas = []

        if self.counter > 0 and self.counter % frame_skip == 0:
            self.api_response = []
            image = frame.copy()
            person_faces = []
            self.drawing_response = video_util.get_default_display_frame()

            model_lst = model.model

            face_model_idx = [index for index, obj in enumerate(
                model_lst) if obj.name == 'gender_face']
            age_model_idx = [index for index, obj in enumerate(
                model_lst) if obj.name == 'gender_age']
            gender_model_idx = [index for index, obj in enumerate(
                model_lst) if obj.name == 'gender_gender']

            face_model = model_lst[int(face_model_idx[0])]
            gender_model = model_lst[int(age_model_idx[0])]
            age_model = model_lst[int(gender_model_idx[0])]

            self.face_model = face_model.load()
            self.gender_model = gender_model.load()
            self.age_model = age_model.load()

            detected_faces = self._detect_face_image(image, face_model.load())

            for face in detected_faces:
                rectangle_position = []
                text_position = []

                xmin, ymin, xmax, ymax = face[0], face[1], face[2], face[3]
                crop_face = image[int(ymin):int(ymax), int(xmin):int(xmax)]

                gender = self._gender_face_image(
                    crop_face, gender_model.load())
                age = self._detect_age_image(crop_face, age_model.load())

                bbox = [[(int(xmin), int(ymin))], (int(xmax), int(ymax))]
                person_face = personFace(bbox=bbox, gender=gender, age=age)
                person_faces.append(person_face)

                rectangle_position.append({
                    "start_point": (int(xmin), int(ymin)),
                    "end_point": (int(xmax), int(ymax)),
                    "color": bgr_green,
                    "thickness": 3,
                })

                text_position.append({
                    "text": 'Gender ['+gender+'] Age ['+age+']',
                    "coord":  (int(xmin - 40), int(ymin - 10)),
                    "color": bgr_green,
                    "thickness": 2,
                })

                this.aval_age_gender.append({
                    "gender": str(gender),
                    "age": str(age)
                })

                coord.append({
                    "0": [xmin, ymin],
                    "1": [xmax, ymax]
                })

                text += 'Gender:['+gender+'] Age:['+age+']'

                self.drawing_response["rectangle"] += rectangle_position
                self.drawing_response["text"] += text_position

                logging.info(person_face)

        if this.aval_age_gender and coord:
            raw_datas.append(
                api_helpers.get_payload_struct(
                        "age-and-gender",
                        coord,
                        str(text),
                        {"age_gender": this.aval_age_gender}
                    )
                )

        if this.age_gender == "" or this.age_gender != str(this.aval_age_gender) and text != "":

            if coord:
                logging.info("Gender Age Changed")
                self.api_response.append(
                    api_helpers.get_payload_struct(
                        "age-and-gender",
                        coord,
                        str(text),
                        {"age_gender": this.aval_age_gender}
                    )
                )
                this.age_gender = str(this.aval_age_gender)

        else:
            self.api_response = []

        return self.drawing_response, self.api_response, raw_datas

    def _detect_face_image(self, image, model):
        return find_faces(image, model)

    def _detect_age_image(self, image, model):
        return find_age(image, model)

    def _gender_face_image(self, image, model):
        return find_gender(image, model)
