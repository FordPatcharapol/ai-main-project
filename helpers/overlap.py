import cv2
import numpy as np


def isOverlap(rois, bbox):

    small_xmin, small_ymin, small_xmax, small_ymax = int(
        bbox["0"][0]), int(bbox["0"][1]), int(bbox["1"][0]), int(bbox["1"][1])

    center_x, center_y = (
        small_xmax+small_xmin)//2, (small_ymax+small_ymin)//2

    if type(rois) is dict:
        big_xmin, big_ymin, big_xmax, big_ymax = int(rois["0"][0]), int(
            rois["0"][1]), int(rois["1"][0]), int(rois["1"][1])
    else:
        big_xmin, big_ymin, big_xmax, big_ymax = rois[0], rois[1], rois[0]+rois[2], rois[1]+rois[3]

    results = cv2.pointPolygonTest(np.array([(big_xmin, big_ymin), (big_xmin, big_ymax),
                                             (big_xmax, big_ymax), (big_xmax, big_ymin)],
                                            np.int32), (center_x, center_y), False)
    if results >= 0:
        return True

    return False
