import ast
from datetime import datetime, timezone
import pydash

from helpers.overlap import isOverlap
from helpers.mode import Mode
from helpers.capture import capture_frame


class ClassSlotContainer:
    def __init__(self):
        self.slots = {}
        self.position = None
        self.send_api = None

    def createSlot(self, index, rois, model_enable, frame_skip):

        slot_name = "slot_" + str(index+1)
        if len(rois) > 4:
            slot_name = rois[4]

        self.slots[slot_name] = ClassSlotItem(slot_name, rois, frame_skip)
        self.slots[slot_name].createList(model_enable)
        self.slots[slot_name].createPosition()

    def getDefaultApiStruct(self, payload):
        return {
            "position": self.position,
            "class": "parking-time",
            "value": {
                "parking_time": payload
            }
        }

    def getPayloadStruct(self, detected_data, name, entry_time, exit_time, frame_entry):
        return {
            "detected_data": detected_data,
            "slot_name": name,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "frame_entry": frame_entry
        }

    def addResult(self, raw_datas, frame):
        response = []
        payload = []
        self.position = []
        self.send_api = None

        for raw_data in raw_datas:

            class_name = pydash.get(raw_data, 'class', [])
            positions = pydash.get(raw_data, 'position', [])
            value = pydash.get(raw_data, 'value', [])

            if positions == [] or class_name == 'space-detection':
                continue

            frame_data = {
                "position": positions,
                "class_name": class_name,
                "value": value
            }

            for _, cur_slot in self.slots.items():

                if cur_slot.addAnalyticResult(frame_data, frame):
                    self.send_api = True

        for _, cur_slot in self.slots.items():

            self.position.append(cur_slot.position)
            payload_struct = self.getPayloadStruct(cur_slot.detected_data,
                                                   cur_slot.name, cur_slot.entry_time, cur_slot.exit_time, cur_slot.frame_entry)
            cur_slot.frame_entry = ""

            if cur_slot.setExit():
                cur_slot.clearValue()

            payload.append(payload_struct)

        if self.send_api:
            response = self.getDefaultApiStruct(payload)

        return response


class ClassSlotItem:
    def __init__(self, name, rois, frame_skip):
        self.name = name
        self.rois = rois
        self.frame_skip = frame_skip
        self.position = None
        self.model_lists = {}
        self.detected_data = []
        self.entry_time = ""
        self.exit_time = ""
        self.prev_plate = None
        self.current_plate = None
        self.counter = 0
        self.mode_province = None
        self.frame_entry = ""

    def addAnalyticResult(self, frame_data, frame):
        positions = frame_data["position"]
        class_name = frame_data["class_name"]
        value = frame_data["value"]

        self.counter += 1

        for i, _ in enumerate(positions):

            if not self.isInSlot(positions[i]):
                continue

            self.counter = 0

            if self.insertValue(class_name, value, i):

                if class_name != "lpr-dl":
                    continue

                if self.setEntry():
                    self.capture(frame)
                    return True

                if self.setExit():
                    return True

        if self.counter > ((self.frame_skip)*2):
            self.counter = 0

            if self.addEmpty():
                if self.setExit():
                    return True

        return None

    def createPosition(self):

        self.position = {
            "0": [self.rois[0], self.rois[1]],
            "1": [self.rois[0] + self.rois[2], self.rois[1] + self.rois[3]]
        }

    def createList(self, model_enable):

        for model in model_enable:
            if model == "parking_space":
                continue
            self.model_lists[model] = Mode(model, 11)

            if model == "lpr_dl":
                self.model_lists["province"] = Mode("province", 5)

    def isInSlot(self, bbox):
        return isOverlap(self.rois, bbox)

    def insertValue(self, class_name, values, index):
        mode = None

        for detected_data in self.detected_data:
            if detected_data["class"] == class_name:
                self.detected_data.remove(detected_data)

        if class_name == "car-brand":

            detected = values["car_brand"][index]
            self.model_lists["car_brand"].addList(str(detected))
            mode = self.model_lists["car_brand"].getMode()

        elif class_name == "car-model":

            detected = values["car_model"][index]
            self.model_lists["car_model"].addList(str(detected))
            mode = self.model_lists["car_model"].getMode()

        elif class_name == "lpr-dl":

            detected = values["license_plates"][index]["plate"]
            self.model_lists["lpr_dl"].addList(str(detected))
            mode = self.model_lists["lpr_dl"].getMode()

            province = values["license_plates"][index]["province"]

            if province != "-":
                self.model_lists["province"].addList(province)
                mode_province = self.model_lists["province"].getMode()
                self.mode_province = mode_province

        if mode:

            if class_name != "lpr-dl":
                mode = ast.literal_eval(mode)

            if class_name == "lpr-dl":

                self.current_plate = mode

                # if car exit
                if self.prev_plate is not None and self.current_plate != self.prev_plate:
                    mode = {'plate': self.prev_plate,
                            'province': self.mode_province}

                else:
                    mode = {'plate': mode, 'province': self.mode_province}

            self.detected_data.append({
                "class": class_name,
                "value": mode
            })

            return True

        return False

    def setEntry(self):

        if self.prev_plate is None and self.current_plate is not None:
            self.entry_time = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            self.prev_plate = self.current_plate
            return True

        return False

    def setExit(self):

        if self.current_plate != self.prev_plate:
            self.exit_time = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            return True

        return False

    def clearValue(self):
        self.detected_data = []
        self.entry_time = ""
        self.exit_time = ""
        self.prev_plate = None
        self.current_plate = None
        self.mode_province = None
        self.frame_entry = ""

        for _, model_list in self.model_lists.items():
            model_list.clearList()

    def addEmpty(self):

        if self.prev_plate is None:
            return False

        for detected_data in self.detected_data:
            if detected_data["class"] == "lpr-dl":
                self.detected_data.remove(detected_data)

        self.model_lists["lpr_dl"].addList('empty')
        mode = self.model_lists["lpr_dl"].getMode()

        if mode:

            self.current_plate = mode

            if self.prev_plate is not None and self.current_plate != self.prev_plate:
                self.detected_data.append({
                    "class": "lpr-dl",
                    "value": {'plate': self.prev_plate, 'province': self.mode_province}})

                return True

        return False

    def capture(self, frame):
        capture_path = "./ml_postprocess/parking_cal/capture/"+str(self.name)+"_"+str(self.entry_time)+"_"+str(self.current_plate)+".jpg"
        self.frame_entry = capture_frame(frame, self.position, capture_path)