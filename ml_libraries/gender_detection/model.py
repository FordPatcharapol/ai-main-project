import cv2
from ultralytics import YOLO


def load_face_model(model_path):
    model = YOLO(model_path)
    return model


def load_age_model(prototxt_path, caffe_path):
    # return cv2.dnn.readNetFromCaffe(prototxt_path, caffe_path)
    return cv2.dnn.readNet(caffe_path, prototxt_path)


def load_gender_model(prototxt_path, caffe_path):
    # return cv2.dnn.readNetFromCaffe(prototxt_path, caffe_path)
    return cv2.dnn.readNet(caffe_path, prototxt_path)
