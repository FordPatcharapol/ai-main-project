from ml_libraries.lpr_img_processing.loader import LprOcrModelLoader
from ml_libraries.lpr_deep_learning.loader import LprDlModelLoader
from ml_libraries.gender_detection.loader import GenderModelLoader
from ml_libraries.parking_space.loader import ParkingSpaceModelLoader
from ml_libraries.car_brand_detection.loader import CarBrandModelLoader
from ml_libraries.object_counting.loader import ObjectCountingModelLoader
from ml_libraries.car_model_detection.loader import CarModelModelLoader
from ml_libraries.emotion_detection.loader import EmotionModelLoader
from ml_libraries.multi_cam_tracking.loader import MultiCameraModelLoader
from ml_libraries.frauding_detection.loader import FraudModelLoader
from ml_libraries.face_recognition.loader import FaceRecogitionModelLoader


class LibraryLoader:
    def __init__(self) -> None:
        self.lpr_ocr_model = LprOcrModelLoader()
        self.lpr_dl_model = LprDlModelLoader()
        self.parking_space_model = ParkingSpaceModelLoader()
        self.car_brand_model = CarBrandModelLoader()
        self.object_counting_model = ObjectCountingModelLoader()
        self.gender_detection_model = GenderModelLoader()
        self.car_model_model = CarModelModelLoader()
        self.emotion_model = EmotionModelLoader()
        self.multi_camera_model = MultiCameraModelLoader()
        self.frauding_detection_model = FraudModelLoader()
        self.face_recognition_model = FaceRecogitionModelLoader()

    def get_lpr_ocr_model(self):
        return self.lpr_ocr_model

    def get_lpr_dl_model(self):
        return self.lpr_dl_model

    def get_parking_space_model(self):
        return self.parking_space_model

    def get_car_brand_model(self):
        return self.car_brand_model

    def get_object_counting_model(self):
        return self.object_counting_model

    def get_gender_detection_model(self):
        return self.gender_detection_model

    def get_car_model_model(self):
        return self.car_model_model

    def get_emotion_model(self):
        return self.emotion_model

    def get_multi_camera_model(self):
        return self.multi_camera_model

    def get_frauding_detection_model(self):
        return self.frauding_detection_model

    def get_face_recognition_model(self):
        return self.face_recognition_model
