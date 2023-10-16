import cv2
import random
import torchreid
import sys

from ultralytics import YOLO
from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.video import VideoUitls
from ml_libraries.object_counting.sort import *

# logger
logging = get_logger(__name__)
api_helpers = ApiHelper()

mot_tracker = Sort()

# package parameter
this = sys.modules[__name__]
video_util = VideoUitls()
bgr_green = (0, 255, 0)

# init
this.obj_reid_lst = {}
this.obj_reid_db = {}
this.object_lst = {}
this.life_frame_threshold = 5
this.died_frame_threshold = 0
this.prev_ids = ''

extractor = torchreid.utils.FeatureExtractor(
    model_name="resnet50", device="cpu", verbose=False
)


def random_color_generator():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def load_detect_model(model_path):
    model = YOLO(model_path)

    return model


def find_object(image, model, classes):
    results = model(image, verbose=False, conf=0.75, classes=classes)
    lst_detections = results[0].boxes.xyxy.tolist()

    return np.array(lst_detections)


def update_tracker(object_lst):
    return mot_tracker.update(object_lst)


def object_searching(track_bbs_ids, frame):
    draw_id_lst = []
    current_id_lst = {}
    rectangle_position = []
    text_position = []
    api_response = []
    text = ""
    coord = []
    aval_tracking = []

    this.object_lst, this.prev_ids
    drawing_response = video_util.get_default_display_frame()

    for idx in range(len(track_bbs_ids)):
        coords = track_bbs_ids.tolist()[idx]
        xmin, ymin, xmax, ymax, object_id = int(coords[0]), int(
            coords[1]), int(coords[2]), int(coords[3]), int(coords[4])

        current_id_lst[str(object_id)] = {
            'bbox': [xmin, ymin, xmax, ymax],
        }

     # create begin obj
    for obj in current_id_lst:
        if not str(obj) in this.object_lst:
            this.object_lst[str(obj)] = {
                'life': 0,
                'bbox': current_id_lst[str(obj)]['bbox'],
            }

    # update life and check is real obj
    for obj in this.object_lst:

        if not str(obj) in sorted(current_id_lst) and this.object_lst[obj]['life'] >= this.life_frame_threshold:
            # print('Life down')
            this.object_lst[obj]['life'] = 0
            continue

        if not str(obj) in sorted(current_id_lst) and this.object_lst[obj]['life'] < this.life_frame_threshold:
            # print('Life down')
            this.object_lst[obj]['life'] -= 1
            continue

        if str(obj) in sorted(current_id_lst):
            this.object_lst[obj]['bbox'] = current_id_lst[obj]['bbox']

        if this.object_lst[obj]['life'] < this.life_frame_threshold:
            # print('Life addd')
            this.object_lst[obj]['life'] += 1

        if this.object_lst[obj]['life'] >= this.life_frame_threshold:
            # print('update reid')
            this.obj_reid_lst[str(obj)] = this.object_lst[str(obj)]

    this.obj_reid_lst_copy = this.obj_reid_lst.copy()

    for obj in this.obj_reid_lst_copy:
        if not str(obj) in this.object_lst:
            del this.obj_reid_lst[str(obj)]

    # print('--------------------------')

    this.object_lst = {key: value for key,
                       value in this.object_lst.items() if value['life'] >= 0}

    for obj_reid in this.obj_reid_lst:
        if not 'feature' in this.obj_reid_lst[obj_reid]:
            crop = frame[ymin:ymax, xmin:xmax]
            crop = cv2.resize(crop, (256, 128))
            cur_feature = extractor(crop)[0].cpu().numpy()
            this.obj_reid_lst[obj_reid]['feature'] = [cur_feature]
            this.obj_reid_lst[obj_reid]['status'] = False

    for obj_db in this.obj_reid_db:
        this.obj_reid_db[obj_db]['status'] = False

    for obj_reid in this.obj_reid_lst:
        key = None
        tmp_min_dis = 1000
        source_lst = np.array(this.obj_reid_lst[obj_reid]['feature'])

        result_match = {}
        this.obj_reid_db_copy = this.obj_reid_db.copy()

        for obj_key in this.obj_reid_db_copy:

            if this.obj_reid_db_copy[obj_key]['status']:
                continue

            dis = np.linalg.norm(
                source_lst - this.obj_reid_db_copy[obj_key]['feature'])

            result_match[obj_key] = {
                'id': obj_key,
                'distance': dis
            }

        if not result_match:
            this.obj_reid_db[obj_reid] = this.obj_reid_lst[obj_reid]
            this.obj_reid_db[obj_reid]['color'] = random_color_generator()
            draw_id_lst.append(obj_reid)
            continue

        for result_id in result_match:
            if result_match[result_id]['distance'] < tmp_min_dis:
                key = result_match[result_id]['id']
                tmp_min_dis = result_match[result_id]['distance']

        this.obj_reid_db[key]['bbox'] = this.obj_reid_lst[obj_reid]['bbox']
        this.obj_reid_db[key]['status'] = True
        draw_id_lst.append(key)

        name_id = 'ID: {}'.format(str(key))
        colour = this.obj_reid_db[key]['color']

        rectangle_position.append({
            "start_point": (int(this.obj_reid_db[key]['bbox']
                                [0]), int(this.obj_reid_db[key]['bbox'][1])),
            "end_point": (int(this.obj_reid_db[key]['bbox']
                              [2]), int(this.obj_reid_db[key]['bbox'][3])),
            "color": colour,
            "thickness": 3,
        })

        text_position.append({
            "text": str(name_id),
            "coord":  (int(this.obj_reid_db[key]['bbox']
                       [0]), int(this.obj_reid_db[key]['bbox'][1]) - 10),
            "color": colour,
            "thickness": 2,
        })

        coord.append({
            "0": [int(this.obj_reid_db[key]['bbox'][0]), int(this.obj_reid_db[key]['bbox'][1])],
            "1": [int(this.obj_reid_db[key]['bbox'][2]), int(this.obj_reid_db[key]['bbox'][3])]
        })

        aval_tracking.append({
            "id": str(key)
        })

    drawing_response["rectangle"] += rectangle_position
    drawing_response["text"] += text_position
    text = "Detected ID: " + str(aval_tracking)

    if str(this.prev_ids) == "" or str(this.prev_ids) != str(aval_tracking):
        if aval_tracking is not None:
            logging.info("Tracker Object Changed")
            api_response.append(
                api_helpers.get_payload_struct(
                    "multi-tracking",
                    coord,
                    text,
                    {"multi_tracking": aval_tracking}
                )
            )
            this.prev_ids = str(aval_tracking)

    else:
        api_response = []

    return drawing_response, api_response
