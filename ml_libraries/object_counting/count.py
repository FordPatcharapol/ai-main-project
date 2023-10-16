from ultralytics import YOLO
from ml_libraries.object_counting.sort import *
from helpers.video import VideoUitls


class ObjectTrack:
    def __init__(self, id) -> None:
        self.id = id
        self.life = 0
        self.status = False


# init
obj_track_lst = []
obj_track_lst_tmp = {}
obj_tracked_lst = {}
obj_track_count_lst = []
prev_obj = []
total_object = 0
life_frame_threshold = 10
died_frame_threshold = 0

mot_tracker = Sort()

video_util = VideoUitls()
bgr_green = (0, 255, 0)


def load_detect_model(model_path):
    model = YOLO(model_path)

    return model


def find_object(image, model, classes):
    results = model(image, verbose=False, conf=0.6, classes=classes)
    lst_class = results[0].boxes.cls.tolist()
    lst_class_name = []

    for cls in lst_class:
        lst_class_name.append(results[0].names[int(cls)])

    lst_detections = results[0].boxes.xyxy.tolist()

    return np.array(lst_detections), lst_class_name


def find_cur_object(image, model, classes):
    results = model(image, verbose=False, conf=0.6, classes=classes)

    lst_class = results[0].boxes.cls.tolist()
    lst_detections = results[0].boxes.xyxy.tolist()
    lst_class_name = []
    drawing_response = video_util.get_default_display_frame()

    for cls in lst_class:
        lst_class_name.append({
            'type': results[0].names[int(cls)]
        })

    for index, obj in enumerate(lst_detections):

        xmin, ymin, xmax, ymax = int(obj[0]), int(
            obj[1]), int(obj[2]), int(obj[3])

        rectangle_position = []
        text_position = []

        rectangle_position.append({
            "start_point": (int(xmin), int(ymin)),
            "end_point": (int(xmax), int(ymax)),
            "color": bgr_green,
            "thickness": 3,
        })

        text_position.append({
            "text": 'ID: ' + str(index),
            "coord":  (int(xmin), int(ymin - 10)),
            "color": bgr_green,
            "thickness": 2,
        })

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    return np.array(lst_detections), drawing_response, lst_detections, lst_class_name


def update_tracker(object_lst):
    return mot_tracker.update(object_lst)


def update_counter(tracker, object_lst, classes_lst):
    current_id_lst = []
    current_pos_lst = []
    global total_object
    global obj_track_count_lst
    global obj_tracked_lst

    drawing_response = video_util.get_default_display_frame()

    for row in tracker:
        rectangle_position = []
        text_position = []

        xmin, ymin, xmax, ymax, object_id = int(row[0]), int(
            row[1]), int(row[2]), int(row[3]), int(row[4])

        current_id_lst.append(object_id)

        # class_name = find_class_match(object_lst, [xmin, ymin, xmax, ymax], classes_lst)
        best_matches = []
        iou_scores = [calculate_iou(
            [xmin, ymin, xmax, ymax], object_box) for object_box in object_lst]
        best_match_index = np.argmax(iou_scores)
        best_matches.append(best_match_index)

        obj_track_lst_tmp[str(object_id)] = {
            "bbox": [xmin, ymin, xmax, ymax],
            "class": classes_lst[best_match_index]
        }

        current_pos_lst.append([
            xmin, ymin, xmax, ymax
        ])

        rectangle_position.append({
            "start_point": (int(xmin), int(ymin)),
            "end_point": (int(xmax), int(ymax)),
            "color": bgr_green,
            "thickness": 3,
        })

        text_position.append({
            "text": 'ID: ' + str(object_id),
            "coord":  (int(xmin), int(ymin - 10)),
            "color": bgr_green,
            "thickness": 2,
        })

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    for id in current_id_lst:
        matching_obj = next(
            (obj for obj in obj_track_lst if obj.id == id), None)

        if matching_obj is None:
            new_object = ObjectTrack(id)
            new_object.life += 1
            obj_track_lst.append(new_object)
            continue

        if matching_obj.status:
            continue

        if matching_obj.life >= life_frame_threshold and matching_obj.status is False:
            total_object += 1
            obj_tracked_lst[str(matching_obj.id)
                            ] = obj_track_lst_tmp[str(matching_obj.id)]
            matching_obj.status = True
        else:
            matching_obj.life += 1

    for obj in obj_track_lst:
        if obj.id not in current_id_lst:
            obj.life -= 1

        if obj.life <= died_frame_threshold:
            obj_track_lst.remove(obj)

    for obj in obj_tracked_lst:

        if int(obj) not in current_id_lst:
            continue

        if obj in prev_obj:
            continue

        type_obj = obj_track_lst_tmp[obj]["class"]
        is_find = False

        prev_obj.append(obj)

        for obj_track_count in obj_track_count_lst:
            if obj_track_count["type"] == type_obj:
                is_find = True
                obj_track_count["total"] += 1

        if not is_find:
            obj_track_count_lst.append({
                "type": type_obj,
                "total": 1
            })
            continue

    return total_object, drawing_response, current_pos_lst, obj_track_count_lst


def calculate_iou(boxA, boxB):
    # Calculate the intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate intersection area
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate IOU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou
