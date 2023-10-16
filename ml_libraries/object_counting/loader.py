import sys
import numpy as np

from ml_libraries.object_counting.count import load_detect_model, find_object, update_tracker, update_counter, find_cur_object, calculate_iou
from helpers.model import ModelLoader
from helpers.video import VideoUitls
from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.url import read_path

# logger
logging = get_logger(__name__)
api_helpers = ApiHelper()

# package parameter
this = sys.modules[__name__]
this.prev_ids = ""
this.aval_count = []
this.ids_lst = []

video_util = VideoUitls()


class ObjectCountingModelLoader:
    def __init__(self) -> None:
        self.loaded_model = None
        self.counter = 0
        self.api_response = []
        self.drawing_response = video_util.get_default_display_frame()

    def model_loader(self, source_config, analytic_key, analytic_value):
        _ = source_config
        analytic_value['detect_model_path'] = read_path(
            analytic_value['detect_model_path'], "object_counting", "yolov8n.pt")

        model_loaded = load_detect_model(
            model_path=analytic_value['detect_model_path'])
        new_model = ModelLoader(analytic_key, model_loaded)

        return new_model

    def extract_frame(self, frame, model, analytics):
        frame_skip = analytics["object_counting"]["frame_skip"]
        classes = analytics["object_counting"]["class"]
        self.counter += 1
        self.api_response = []
        coord = []
        text = ""
        cur_total_obj = None
        raw_datas = []

        if self.counter > 0 and self.counter % frame_skip == 0:
            image = frame.copy()

            self.drawing_response = video_util.get_default_display_frame()

            if self.loaded_model is None:
                self.loaded_model = model.load()

            # curr frame

            object_lst, drawing_response, current_pos_lst, obj_track_count_lst = find_cur_object(
                image, self.loaded_model, classes)

            cur_total_obj = len(obj_track_count_lst)

            # if len(object_lst) == 0:
            #     return self.drawing_response, self.api_response, raw_datas

            # cont frame

            object_lst, classes_lst = find_object(
                image, self.loaded_model, classes)

            if len(object_lst) == 0:
                raw_datas = self.empty_raw_data(raw_datas)
                return self.drawing_response, self.api_response, raw_datas

            tracker_lst = update_tracker(object_lst)

            raw_lst = tracker_lst.tolist().copy()
            raw_datas = self.prepare_raw_data(object_lst, classes_lst, raw_lst, raw_datas)

            # cur_total_obj, drawing_response, current_pos_lst, obj_track_count_lst = update_counter(
            #     tracker_lst, object_lst, classes_lst)

            # end cont frame

            this.aval_count = obj_track_count_lst

            for cur_pos in current_pos_lst:
                coord.append({
                    "0": [cur_pos[0], cur_pos[1]],
                    "1": [cur_pos[2], cur_pos[3]]
                })

            text = 'Total Object: ' + str(obj_track_count_lst)

            logging.info("Total object: %s", cur_total_obj)

            if cur_total_obj > 0:
                self.drawing_response = drawing_response

        # print('first this.prev_ids', self.api_response, 'this.ids_lst', this.ids_lst)

        if this.prev_ids == "" or this.prev_ids != str(cur_total_obj):
            if cur_total_obj is not None:
                logging.info("Total Object Changed")
                self.api_response.append(
                    api_helpers.get_payload_struct(
                        "object-counting",
                        coord,
                        text,
                        {"object_count": this.aval_count}
                    )
                )
                this.prev_ids = str(cur_total_obj)

        else:
            self.api_response = []

        return self.drawing_response, self.api_response, raw_datas

    def prepare_raw_data(self, object_lst, classes_lst, raw_lst, raw_datas):

        raw_coord = []
        id_name = []

        for row in raw_lst:

            xmin, ymin, xmax, ymax, object_id = int(row[0]), int(
                row[1]), int(row[2]), int(row[3]), int(row[4])

            iou_scores = [calculate_iou(
                [xmin, ymin, xmax, ymax], object_box) for object_box in object_lst]
            best_match_index = np.argmax(iou_scores)

            id_name.append({
                "id_name": "id " + str(object_id),
                "type": classes_lst[best_match_index]
            })

            raw_coord.append({
                "0": [xmin, ymin],
                "1": [xmax, ymax]
            })


        text = 'Total Object: ' + str(len(id_name))

        if id_name and raw_coord :

            raw_datas.append(
                api_helpers.get_payload_struct(
                        "object-counting",
                        raw_coord,
                        text,
                        {"object_count": id_name}
                    )
                )

        return raw_datas

    def empty_raw_data(self, raw_datas):
        raw_coord = [{
                "0": [0, 0],
                "1": [0, 0]
            }]
        text = 'Total Object: 0'
        id_name = [{
                "id_name": "id 0",
                "type": "empty"
            }]

        raw_datas.append(
            api_helpers.get_payload_struct(
                    "object-counting",
                    raw_coord,
                    text,
                    {"object_count": id_name}
                )
            )
        return raw_datas
