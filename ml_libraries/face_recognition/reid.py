import cv2
import face_recognition
import json
import numpy as np
import pydash
import sys

from ultralytics import YOLO
from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.mode import Mode
from ml_libraries.face_recognition.sqlite import create_table, insert_varible_into_table, getFeatureInfo, update_db

# logger
api_helpers = ApiHelper()
logging = get_logger(__name__)

# package parameter
this = sys.modules[__name__]
this.id = []
this.names = []
this.surnames = []
this.nickname = []
this.ages = []
this.genders = []
this.face_id = []
this.people_datas = None
this.detected_face = []
this.prev_detected_face_feature = []


class FaceInfo:
    def __init__(self, face_vec, pos):
        self.face_vec = face_vec
        self.position = pos
        self.id = Mode('id', 5)
        self.firstname = Mode('firstname', 5)
        self.surname = Mode('surname', 5)
        self.nickname = Mode('nickname', 5)
        self.age = Mode('age', 5)
        self.gender = Mode('gender', 5)
        self.life = 3
        self.threshold_life = 5
        self.threshold_died = 0

    def __str__(self):
        return '[' + str(self.id.getMode()) + '] Name: ' + str(self.nickname.getMode()) + '(' + str(self.life) + ')'


# Model
def load_face_model(model_path):
    model = YOLO(model_path)

    return model


def face_location(img, model):
    face = model(source=img, conf=0.6, verbose=False)
    face_bboxs = face[0].boxes.xyxy.tolist()

    return face_bboxs


def load_to_sqlite():
    if this.people_datas_status:
        create_table()

        for person in this.people_datas:
            id = person['id']
            firstname = "".join(person['firstname'].split())
            surname = "".join(person['surname'].split())
            nickname = "".join(person['nickname'].split())
            age = person['age']
            gender = "".join(person['gender'].split())
            face_id = json.dumps(person['face_id'])

            if face_id is None or len(face_id) == 0:
                face_id = json.dumps({})

            insert_varible_into_table(
                id, firstname, surname, nickname, age, gender, face_id)

    this.id, this.names, this.surnames, this.nickname, this.ages, this.genders, this.face_id = getFeatureInfo()


def load_encoding_from_db():
    this.people_datas = api_helpers.get_people_info()

    if this.people_datas != []:
        this.people_datas_status = True
        this.people_datas = this.people_datas["data"]
        logging.info('Done to load feature data from API')
    else:
        this.people_datas_status = False
        logging.info('Can not load feature data from API !!')

    update_db(this.people_datas_status)


def detect_known_faces(frame, model):

    face_id = []
    face_names = []
    face_surnames = []
    face_nicknames = []
    face_genders = []
    face_ages = []
    face_locations = []

    if not this.face_id:
        return None, None, None, None, None, None, None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_pos = face_location(frame, model)

    for face_pos in faces_pos:
        xmin, ymin, xmax, ymax = int(face_pos[0]), int(
            face_pos[1]), int(face_pos[2]), int(face_pos[3])
        face_locations.append((ymin, xmax, ymax, xmin))

    face_encodings = face_recognition.face_encodings(
        rgb_frame, face_locations)

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            this.face_id, face_encoding)
        nickname = "Unknown"

        face_distances = face_recognition.face_distance(
            this.face_id, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            nickname = this.nickname[best_match_index]
        face_nicknames.append(nickname)

        if nickname != "Unknown":
            face_id.append(this.id[best_match_index])
            face_names.append(this.names[best_match_index])
            face_surnames.append(this.surnames[best_match_index])
            face_genders.append(this.genders[best_match_index])
            face_ages.append(this.ages[best_match_index])
        else:
            face_id.append('Unknown')
            face_names.append('Unknown')
            face_surnames.append('Unknown')
            face_genders.append('Unknown')
            face_ages.append('Unknown')

    faces_pos, face_id, face_names, face_surnames, face_nicknames, face_genders, face_ages = face_reid_mode(zip(faces_pos, face_id, face_names, face_surnames,
                                                                                                                face_nicknames, face_genders, face_ages, face_encodings))

    return faces_pos, face_id, face_names, face_surnames, face_nicknames, face_genders, face_ages


def face_reid_mode(faces):
    detected_face_feature = []
    prev_detected = []
    faces_pos = []
    face_id = []
    face_names = []
    face_surnames = []
    face_nicknames = []
    face_genders = []
    face_ages = []

    for face_obj in this.detected_face:
        face_obj.life -= 1
        prev_detected.append(str(face_obj.id) + str(face_obj.firstname) + str(
            face_obj.surname) + str(face_obj.nickname) + str(face_obj.gender) + str(face_obj.age))
        detected_face_feature.append(face_obj.face_vec.tolist())

    for face_loc, id, name, sur, nickname, gender, age, face_feature in faces:
        best_match_index = None

        refer = str(id) + str(name) + str(sur) + \
            str(nickname) + str(gender) + str(age)

        if this.detected_face == []:
            new_face = FaceInfo(face_feature, face_loc)
            new_face.id.addList(str(id))
            new_face.firstname.addList(str(name))
            new_face.surname.addList(str(sur))
            new_face.nickname.addList(str(nickname))
            new_face.age.addList(str(age))
            new_face.gender.addList(str(gender))
            this.detected_face.append(new_face)
            prev_detected.append(refer)
            continue

        if detected_face_feature:

            matches = face_recognition.compare_faces(
                detected_face_feature, face_feature)
            face_distances = face_recognition.face_distance(
                detected_face_feature, face_feature)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                this.detected_face[best_match_index].face_vec = face_feature
                this.detected_face[best_match_index].position = face_loc
                this.detected_face[best_match_index].id.addList(str(id))
                this.detected_face[best_match_index].firstname.addList(
                    str(name))
                this.detected_face[best_match_index].surname.addList(str(sur))
                this.detected_face[best_match_index].nickname.addList(
                    str(nickname))
                this.detected_face[best_match_index].age.addList(str(age))
                this.detected_face[best_match_index].gender.addList(
                    str(gender))

                if this.detected_face[best_match_index].life < this.detected_face[best_match_index].threshold_life:
                    this.detected_face[best_match_index].life += 2

            elif not matches[best_match_index] and refer not in prev_detected:
                new_face = FaceInfo(face_feature, face_loc)
                new_face.id.addList(str(id))
                new_face.firstname.addList(str(name))
                new_face.surname.addList(str(sur))
                new_face.nickname.addList(str(nickname))
                new_face.age.addList(str(age))
                new_face.gender.addList(str(gender))
                this.detected_face.append(new_face)
                prev_detected.append(refer)

    indices_to_remove = []

    for index, face_obj in enumerate(this.detected_face):
        if face_obj.life <= face_obj.threshold_died:
            indices_to_remove.append(index)

    for index in reversed(indices_to_remove):
        del this.detected_face[index]

    if not this.detected_face:
        return None, None, None, None, None, None, None

    for face_obj in this.detected_face:
        if face_obj.id.getMode() is None:
            continue

        faces_pos.append(face_obj.position)
        face_id.append(face_obj.id.getMode())
        face_names.append(face_obj.firstname.getMode())
        face_surnames.append(face_obj.surname.getMode())
        face_nicknames.append(face_obj.nickname.getMode())
        face_genders.append(face_obj.gender.getMode())
        face_ages.append(face_obj.age.getMode())

    return faces_pos, face_id, face_names, face_surnames, face_nicknames, face_genders, face_ages
