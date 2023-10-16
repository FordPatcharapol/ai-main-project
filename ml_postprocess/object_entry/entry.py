import ast
from datetime import datetime, timezone
import pydash

from helpers.overlap import isOverlap
from helpers.mode import Mode
from helpers.capture import capture_frame


class ClassEntranceContainer:
    def __init__(self):
        self.object = {}
        self.slot = {}
        self.model_enable = []

    def createObject(self, class_name, positions, value, frame):

        for i, _ in enumerate(positions):
            id_name = value["object_count"][i]["id_name"]

            if id_name in self.object:
                self.object[id_name].position = positions[i]
                self.object[id_name].isEntry(frame)
                self.object[id_name].counter = 0
                continue

            type_id = value["object_count"][i]["type"]
            self.object[id_name] = ClassObjectItem(id_name, type_id, positions[i],
                                                   class_name, pydash.clone_deep(self.slot))
            self.object[id_name].createList(self.model_enable)

    def createEntrace(self, rois, model_enable):

        for i, roi in enumerate(rois):

            slot_name = "door_" + str(i+1)
            if len(roi) > 4:
                slot_name = roi[4]

            self.slot[slot_name] = roi

        self.model_enable = model_enable

    def getDefaultApiStruct(self, position, payload):
        return {
            "position": position,
            "class": "object-entry",
            "value": {
                "object_entry": payload
            }
        }

    def getPayloadStruct(self, detected_data, class_name, type_id, slot_name, entry_time, frame_entry):
        return {
            "detected_data": detected_data,
            "primary_type": class_name,
            "type": type_id,
            "slot_name": slot_name,
            "entry_time": entry_time,
            "frame_entry": frame_entry
        }

    def addResult(self, raw_datas, frame):

        response = []
        position = []
        payload = []
        exit_object = []

        for raw_data in raw_datas:

            class_name = pydash.get(raw_data, 'class', [])
            positions = pydash.get(raw_data, 'position', [])
            value = pydash.get(raw_data, 'value', [])

            if positions == [] or class_name == 'space-detection':
                continue

            if class_name == "object-counting":
                if value["object_count"][0]["id_name"] == "id 0":
                    for _, cur_object in self.object.items():
                        cur_object.checkOutFrame()
                    continue
                self.createObject(class_name, positions, value, frame)
                continue

            for _, cur_object in self.object.items():
                cur_object.addAnalyticResult(class_name, positions, value)
                cur_object.checkOutFrame()

        for id_name, cur_object in self.object.items():
            if cur_object.exit:
                if cur_object.entry_time is None:
                    exit_object.append(id_name)
                    continue

                payload_struct = self.getPayloadStruct(cur_object.detected_data, cur_object.class_name,
                                                       cur_object.type_id, cur_object.entry_door, cur_object.entry_time, cur_object.frame_entry)
                payload.append(payload_struct)
                position.append(cur_object.position)
                exit_object.append(id_name)

        for id_name in exit_object:
            del self.object[id_name]

        if payload and position:
            response = self.getDefaultApiStruct(position, payload)

        return response


class ClassObjectItem:
    def __init__(self, id_name, type_id, position, class_name, slot):
        self.id_name = id_name
        self.type_id = type_id
        self.position = position
        self.class_name = class_name
        self.slot = slot
        self.slot_list = {}
        self.counter = 0
        self.model_lists = {}
        self.detected_data = []
        self.entry_time = None
        self.entry_door = None
        self.exit = None
        self.frame_entry = ""

    def addAnalyticResult(self, class_name, positions, value):

        for i, _ in enumerate(positions):
            if not isOverlap(self.position, positions[i]):
                continue
            self.insertValue(class_name, value, i)

    def createList(self, model_enable):

        for model in model_enable:
            if model == "parking_space" or model == "object_counting":
                continue

            if model == "gender":
                self.model_lists["age"] = Mode("age", 5)

            self.model_lists[model] = Mode(model, 5)

        for slot_name in self.slot:
            self.slot_list[str(slot_name)] = Mode(str(slot_name), 5)

    def isEntry(self, frame):

        for slot_name, roi in self.slot.items():
            if not isOverlap(roi, self.position):
                continue

            id_name = self.id_name
            self.slot_list[str(slot_name)].addList(str(id_name))
            mode = self.slot_list[str(slot_name)].getMode()

            if mode and self.entry_time is None:
                self.entry_time = datetime.now(
                    timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                self.entry_door = slot_name
                self.capture(roi, frame)

    def insertValue(self, class_name, values, index):
        mode = None

        for detected_data in self.detected_data:
            if detected_data["class"] == class_name:
                self.detected_data.remove(detected_data)

        if class_name == "mood-tone":

            detected = values["cur_emotion"][index]["emotion"]
            self.model_lists["emotion"].addList(str(detected))
            mode = self.model_lists["emotion"].getMode()

            if mode:
                mode = str({'emotion': mode})

        elif class_name == "face-recognition":

            detected = values["peoples"][index]
            self.model_lists["face_recognition"].addList(str(detected))
            mode = self.model_lists["face_recognition"].getMode()

        elif class_name == "age-and-gender":

            age_detected = values["age_gender"][index]["age"]
            self.model_lists["age"].addList(str(age_detected))
            age_mode = self.model_lists["age"].getMode()

            gender_detected = values["age_gender"][index]["gender"]
            self.model_lists["gender"].addList(str(gender_detected))
            gender_mode = self.model_lists["gender"].getMode()

            if age_mode and gender_mode:
                mode = str({'gender': gender_mode, 'age': age_mode})

        if mode:

            mode = ast.literal_eval(mode)

            self.detected_data.append({
                "class": class_name,
                "value": mode
            })

    def checkOutFrame(self):
        self.counter += 1
        if self.counter > 10:
            self.exit = True

    def capture(self, roi, frame):
        capture_path = "./ml_postprocess/object_entry/capture/" + str(self.entry_door)+"_"+str(self.entry_time) + "_"+str(self.id_name)+".jpg"
        self.frame_entry = capture_frame(frame, roi, capture_path)
