import numpy as np

from ultralytics import YOLO

theshold_arm_angle = 40
theshold_arm_angle_mix = 0


class ObjectInfo:
    def __init__(self, bbox, status, time):
        self.bbox = bbox
        # self.keypoints = keypoints
        self.is_thief = status
        self.timestamp = time

def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def get_angle(pointsList):
    pt1, pt2, pt3 = np.array(pointsList[0]), np.array(
        pointsList[1]), np.array(pointsList[2])

    radians = np.arctan2(pt3[1] - pt2[1], pt3[0] - pt2[0]) - \
        np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def load_pose_model(model_path):
    model = YOLO(model_path)
    return model


def get_arms_lst(keypoints):
    left_arm_list = [
        [int(keypoints[5][0]), int(keypoints[5][1])],
        [int(keypoints[7][0]), int(keypoints[7][1])],
        [int(keypoints[9][0]), int(keypoints[9][1])]
    ]

    right_arm_list = [
        [int(keypoints[6][0]), int(keypoints[6][1])],
        [int(keypoints[8][0]), int(keypoints[8][1])],
        [int(keypoints[10][0]), int(keypoints[10][1])]
    ]

    return left_arm_list, right_arm_list


def is_shoplifter(left_arm_list, right_arm_list):
    left_arm_angle = get_angle(left_arm_list)
    right_arm_angle = get_angle(right_arm_list)

    if left_arm_angle < theshold_arm_angle and left_arm_angle >= theshold_arm_angle_mix:
        return True

    if right_arm_angle < theshold_arm_angle and right_arm_angle >= theshold_arm_angle_mix:
        return True

    return False


def is_thief(keypoints):
    left_arm_list, right_arm_list = get_arms_lst(keypoints)
    shoplifter_ststus = is_shoplifter(left_arm_list, right_arm_list)

    return shoplifter_ststus
