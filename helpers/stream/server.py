# Import the required modules
import socket
import cv2
import pickle
import struct  # new

from helpers.logger import get_logger

# logger
logging = get_logger(__name__)


class ServerHelpers:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.addr = None
        self.socket = None
        self.data = b""
        self.payload_size = 0

    def create_connection(self):
        HOST = self.config['server_listen_address']
        PORT = self.config['server_listen_port']

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logging.info('Socket created')

        self.socket.bind((HOST, PORT))
        logging.info('Socket bind complete')

        self.socket.listen(10)
        logging.info('Socket now listening at %s, port %s', HOST, PORT)

        self.conn, self.addr = self.socket.accept()

        self.payload_size = struct.calcsize(">L")
        logging.info("payload_size: %s", format(self.payload_size))

    def listen_frame(self):
        while len(self.data) < self.payload_size:
            self.data += self.conn.recv(4096)
            if not self.data:
                cv2.destroyAllWindows()
                self.conn, self.addr = self.socket.accept()

                continue

        # receive image row data form client socket
        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        while len(self.data) < msg_size:
            self.data += self.conn.recv(4096)

        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # unpack image using pickle
        frame = pickle.loads(
            frame_data, fix_imports=True, encoding="bytes")

        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # cv2.imshow('server', frame)

        return frame
