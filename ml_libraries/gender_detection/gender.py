import cv2
import numpy as np

# init
indexes = np.array([i for i in range(0, 101)])
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


class personFace:
    def __init__(self, bbox, gender, age):
        self.bbox = bbox
        self.gender = gender
        self.age = age

    def __str__(self):
        return f'Position: {self.bbox},\t Age: {self.age},\t Gender: {self.gender}'

    def get_face(self):
        return {self.gender, self.bbox}


def find_faces(image, model):
    # Inference
    results = model.predict(image, conf=0.7, verbose=False)
    objects = results[0].boxes.data.tolist()

    return objects


def find_gender(image, model):
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    model.setInput(blob)
    genderPreds = model.forward()
    gender = genderList[genderPreds[0].argmax()]
    # print(f'Gender: {gender}')

    return gender


def find_age(image, model):
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    model.setInput(blob)
    agePreds = model.forward()
    age = ageList[agePreds[0].argmax()]
    # print(f'Age: {age[1:-1]} years')

    return age[1:-1]
