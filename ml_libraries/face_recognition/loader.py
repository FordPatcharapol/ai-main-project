import sys
import time

from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.video import VideoUitls
from helpers.model import ModelLoader
from helpers.url import read_path

from ml_libraries.face_recognition.reid import load_face_model, load_encoding_from_db, load_to_sqlite, detect_known_faces


# package parameter
this = sys.modules[__name__]
this.aval_reid = []
this.prev_reid = ''
this.cur_people = []

# logger
logging = get_logger(__name__)
video_util = VideoUitls()
api_helpers = ApiHelper()


class FaceRecogitionModelLoader:
    def __init__(self) -> None:
        self.loaded_model = None
        self.counter = 0
        self.drawing_response = video_util.get_default_display_frame()
        self.api_response = []
        self.last_update = int(time.time())
        load_encoding_from_db()
        load_to_sqlite()

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config
        logging.info('Load Face Recognition Model...')

        analytic_value['face_model_path'] = read_path(
            analytic_value['face_model_path'], "face_recognition", "face_detect.pt")

        model_loaded = load_face_model(
            model_path=analytic_value['face_model_path'])
        new_model = ModelLoader(analytic_key, model_loaded)

        return new_model

    def extract_frame(self, frame, model, analytics):
        """Extrack Face ID"""

        period_interval = int(analytics["face_recognition"]["period_interval"])
        frame_skip = analytics["face_recognition"]["frame_skip"]
        color = (0, 255, 0)
        current_time = int(time.time())
        time_diff = current_time - self.last_update

        self.counter += 1

        self.api_response = []
        this.aval_reid = []
        this.cur_people = []
        text = ''
        coord = []

        raw_datas = []
        if self.counter > 0 and self.counter % frame_skip == 0:
            self.drawing_response = video_util.get_default_display_frame()

            face_locations, face_id, face_names, face_surnames, face_nicknames, face_genders, face_ages = detect_known_faces(
                frame, model.load())

            if face_locations is None:
                return [], [], []

            for face_loc, id, name, sur, nickname, gender, age in zip(face_locations, face_id, face_names, face_surnames, face_nicknames, face_genders, face_ages):
                xmin, ymin, xmax, ymax = int(face_loc[0]), int(
                    face_loc[1]), int(face_loc[2]), int(face_loc[3])

                full_name = name + " " + sur

                if full_name in this.cur_people:
                    continue

                if nickname == 'Unknown':
                    color = (0, 0, 255)

                logging.info('Find Re-Identification > [%s] Name: %s %s (%s)-%s-%s years.',
                             id, name, sur, nickname, gender, age)

                self.drawing_response["text"] += [{
                    "text": "[" + str(id) + "]: " + name + ' ' + sur,
                    "coord":  (int(xmin), int(ymin - 20)),
                    "color": color,
                    "thickness": 2,
                }]

                self.drawing_response["rectangle"].append({
                    "start_point": (int(xmin), int(ymin)),
                    "end_point": (int(xmax), int(ymax)),
                    "color": color,
                    "thickness": 3,
                })

                if nickname == 'Unknown':
                    this.aval_reid.append({
                        "nickname": "Unknown",
                        "id": 0,
                        "firstname": "Unknown",
                        "surname": "Unknown",
                        "age": 0,
                        "gender": "other"
                    })

                    coord.append({
                        "0": [xmin, ymin],
                        "1": [xmax, ymax]
                    })

                    text += name + " " + sur + ", "

                    this.cur_people.append(full_name)

                    continue

                this.aval_reid.append({
                    "nickname": nickname,
                    "id": id,
                    "firstname": name,
                    "surname": sur,
                    "age": age,
                    "gender": gender
                })

                coord.append({
                    "0": [xmin, ymin],
                    "1": [xmax, ymax]
                })

                text += name + " " + sur + ", "

                this.cur_people.append(full_name)

            self.counter = 0
            this.cur_people.sort()

            if this.aval_reid and coord:
                raw_datas.append(
                    api_helpers.get_payload_struct(
                        "face-recognition",
                        coord,
                        "reid: [" + str(text) + "]",
                        {"peoples": this.aval_reid}
                    )
                )

            if time_diff > period_interval and this.prev_reid == str(this.cur_people):
                logging.info("Update Re-Identification")
                self.api_response.append(
                    api_helpers.get_payload_struct(
                        "face-recognition",
                        coord,
                        "reid: [" + str(text) + "]",
                        {"peoples": this.aval_reid}
                    )
                )
                this.prev_reid = str(this.cur_people)
                self.last_update = current_time

                return self.drawing_response, self.api_response, raw_datas

            if this.prev_reid == "" or this.prev_reid != str(this.cur_people) and this.aval_reid != []:
                logging.info("Update Re-Identification")
                self.api_response.append(
                    api_helpers.get_payload_struct(
                        "face-recognition",
                        coord,
                        "reid: [" + str(text) + "]",
                        {"peoples": this.aval_reid}
                    )
                )
                this.prev_reid = str(this.cur_people)
                self.last_update = current_time
            else:
                self.api_response = []

        return self.drawing_response, self.api_response, raw_datas
