# importing everything we need
import csv
import math
import cv2
import pydash
import pandas as pd
import numpy as np
import sys

from ultralytics import YOLO

from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.video import VideoUitls


# logger
logging = get_logger(__name__)

video_util = VideoUitls()
api_helpers = ApiHelper()

bgr_green = (0, 255, 0)
bgr_red = (0, 0, 255)
bgr_lime = (128, 255, 0)


# package parameter
this = sys.modules[__name__]
this.prev_slots = ""
this.rois_file = {}
this.class_path = {}
this.x_ratio = None
this.y_ratio = None


def load_parking_model(model_path):
    return YOLO(model_path)


def class_coco(class_path):
    if class_path in this.class_path:
        return pydash.clone_deep(this.class_path[class_path])

    my_file = open(class_path, "r")
    data = my_file.read()
    class_list = data.split("\n")

    this.class_path[class_path] = class_list
    return class_list


def get_rois(rois_file):
    if rois_file in this.rois_file:
        return pydash.clone_deep(this.rois_file[rois_file])

    # getting the spots coordinates into a list
    with open(rois_file, 'r', newline='') as inf:
        csvr = csv.reader(inf)
        rois = list(csvr)

    # converting the values to integer
    rois = [[int(float(j)) if j.isnumeric() else j for j in i] for i in rois]

    this.rois_file[rois_file] = rois
    return rois


def resize_rois(rois, new_size):

    old_width, old_height = 1920, 1080
    new_width = int(new_size.split("x")[0])
    new_height = int(new_size.split("x")[1])

    this.x_ratio = new_width / old_width
    this.y_ratio = new_height / old_height

    for roi in rois:

        roi[0] = int(roi[0] * this.x_ratio)
        roi[1] = int(roi[1] * this.y_ratio)
        roi[2] = int(roi[2] * this.x_ratio)
        roi[3] = int(roi[3] * this.y_ratio)

    return rois


def parking_space(model, frame, class_list, in_rois, class_detect):
    rois = pydash.clone_deep(in_rois)
    drawing_response = video_util.get_default_display_frame()

    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data.tolist()
    px = pd.DataFrame(a).astype("float")

    occupied = []
    empty = []
    cur_slots = []
    coord = []

    dot_position = []

    for _, roww in px.iterrows():

        cls = class_list[int(roww[5])]

        # if 'person' not in cls:
        if len(pydash.intersection([cls], class_detect)) <= 0:
            # if 'car' not in cls and 'person' not in cls:
            continue

        cx_car = int(int(roww[0]) + int(roww[2]))//2
        cy_car = int(int(roww[1]) + int(roww[3]))//2

        dot_position.append({
            "coord": (cx_car, cy_car),
            "radius": 3,
            "color": bgr_red,
            "thickness": -1,
        })

        drawing_response["dot"] += dot_position

    for i, _ in enumerate(rois):
        rectangle_position = []
        text_position = []

        slot_name = "slot_" + str(i+1)
        if len(rois[i]) > 4:
            slot_name = rois[i][4]

        for _, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]

            if len(pydash.intersection([c], class_detect)) <= 0:
                # if 'person' not in c:
                continue

            cx = int(x1+x2)//2
            cy = int(y1+y2)//2

            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)
            # cv2.circle(frame, (cx, cy), 3, bgr_red, -1)

            results = cv2.pointPolygonTest(np.array([(rois[i][0], rois[i][1]), (rois[i][0], rois[i][1]+rois[i][3]), (
                rois[i][0]+rois[i][2], rois[i][1]+rois[i][3]), (rois[i][0]+rois[i][2], rois[i][1])], np.int32), (cx, cy), False)

            if results >= 0 and slot_name not in occupied:
                occupied.append(slot_name)

        rectangle_position_toappend = {
            "start_point": [rois[i][0], rois[i][1]],
            "end_point": [rois[i][0] + rois[i][2], rois[i][1] + rois[i][3]],
            "thickness": 3,
        }

        text_position_toappend = {
            "text": slot_name,
            "coord":  (math.ceil(rois[i][0] + rois[i][2] / 2), rois[i][1] - 10),
            "thickness": 2,
        }
        coord.append({
            "0": rectangle_position_toappend['start_point'],
            "1": rectangle_position_toappend['end_point']
        })

        if slot_name not in occupied:
            cur_slots.append({"name": slot_name, "status": "available"})
            empty.append(slot_name)

            rectangle_position_toappend["color"] = bgr_green
            text_position_toappend["color"] = bgr_green
        else:
            cur_slots.append({"name": slot_name,
                             "status": "unavailable"})
            rectangle_position_toappend["color"] = bgr_red
            text_position_toappend["color"] = bgr_red

        rectangle_position.append(rectangle_position_toappend)
        text_position.append(text_position_toappend)

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    num_aval = str(len(empty))
    aval_spot = str(empty)

    # adding the number of available spots on the shown image
    drawing_response["text"] += [{
        "text": 'Number of Available Spots: ' + num_aval,
        "coord": (int(20*this.x_ratio), int(80*this.y_ratio)),
        "color":  bgr_lime, "thickness": 2,
    }, {
        "text": 'Available Spots: ' + aval_spot,
        "coord": (int(20*this.x_ratio), int(120*this.y_ratio)),
        "color":  bgr_lime, "thickness": 2,
    }]

    # raw_data = None
    # if cur_slots:
    #     raw_data = api_helpers.get_payload_struct(
    #         "space-detection",
    #         coord,
    #         'Available Spots: ' + aval_spot,
    #         {"type": 'car', "slots": cur_slots}
    #     )

    api_response = None
    if this.prev_slots == "" or this.prev_slots != aval_spot:
        logging.info("parking slot changed")
        api_response = api_helpers.get_payload_struct(
            "space-detection",
            coord,
            'Available Spots: ' + aval_spot,
            {"type": 'car', "slots": cur_slots}
        )

        this.prev_slots = aval_spot

    return num_aval, aval_spot, drawing_response, api_response, []
