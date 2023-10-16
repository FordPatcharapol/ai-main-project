import cv2
import os


def capture_frame(frame, position, file_path):

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    if type(position) is dict:
        xmin, ymin, xmax, ymax = int(position["0"][0]), int(
            position["0"][1]), int(position["1"][0]), int(position["1"][1])
    else:
        xmin, ymin, xmax, ymax = position[0], position[1], position[0]+position[2], position[1]+position[3]

    crop_img = frame[ymin:ymax, xmin:xmax]
    new_file_path = file_path.replace(":", "-")
    cv2.imwrite(new_file_path, crop_img)

    frame_encode = str(open(new_file_path, 'rb').read())
    os.remove(new_file_path)

    return frame_encode
