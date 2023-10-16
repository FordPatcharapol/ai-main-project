import pickle
import socket
import struct

import cv2
import imutils

# encode to jpeg format
# encode param image quality 0 to 100. default:95
# if you want to shrink data size, choose low image quality.
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


class ClientHelpers:
    def __init__(self, config):
        self.config = config

        if config['stream_to_server_enable'] is not True:
            return

        self.clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientsocket.connect(
            (config['stream_to_url'], config['stream_to_port']))
        self.clientsocket.settimeout(5)

        self.img_counter = 0

    def send_stream(self, frame):
        if self.config['stream_to_server_enable'] is not True:
            return

        # frame = imutils.resize(frame, width=640)
        _, image = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(image, 0)
        size = len(data)

        if self.img_counter % 10 == 0:
            self.clientsocket.sendall(struct.pack(">L", size) + data)

        self.img_counter += 1
