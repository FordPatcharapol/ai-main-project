""" Video uitility class """
# from PIL import ImageFont, ImageDraw, Image
import cv2
from helpers.config import load_config
from helpers.logger import get_logger
from helpers.stream.server import ServerHelpers


# logger
logging = get_logger(__name__)

# font = ImageFont.truetype("./assets/fonts/Sarun's ThangLuang.ttf", 18)
font_path = "./assets/fonts/Sarun's ThangLuang.ttf"


class VideoUitls:
    def __init__(self):
        source_config, server_config, using_analytics, using_general, _ = load_config()
        self.general = using_general
        self.source = source_config
        self.server = server_config
        self.user = using_analytics
        self.server_helpers = None
        self.frame = None
        self.video = None

    def load_input(self, source, server):
        if source['type'] == 'webcam':
            video = cv2.VideoCapture(int(source['input']))
            video.set(cv2.CAP_PROP_FOURCC,
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

            fps = int(video.get(5))
            logging.info("frame per secs: %s", fps)

            self.video = video
        elif source['type'] == 'video_img':
            video = cv2.VideoCapture(source['input'])
            self.video = video

        elif source['type'] == 'stream':
            self.server_helpers = ServerHelpers(server)
            self.server_helpers.create_connection()
            self.server_helpers.listen_frame()

    def read_frame(self):
        if self.source['type'] == 'webcam' or self.source['type'] == 'video_img':
            return self.video.read()
        if self.source['type'] == 'stream':
            return True, self.server_helpers.listen_frame()

    def set_frame(self, frame):
        self.frame = frame

    def display(self, title):
        self.display_frame(title, self.frame)

    def get_default_display_frame(self):
        return {
            "rectangle": [{"start_point": (), "end_point": (), "color": (),
                           "thickness": 0}],
            "text":  [{"text": "", "coord": (), "color": (), "thickness": 0}],
            "dot":  [{"coord": (), "radius": 0, "color": (), "thickness": 0}],
        }

    def display_frame(self, title, frame):
        """ display_frame: using to display image or video conditionally
           param
           title :
           frame_buffer :
        """

        self.frame = frame
        if self.source["cam_enable"] is True:
            frame = cv2.resize(frame, (854, 480))
            cv2.imshow(title, frame)

    def overlay_frame(self, frame, drawing_responses):
        for drawing_response in drawing_responses:
            if "rectangle" in drawing_response and drawing_response["rectangle"] is not None:
                for drp in drawing_response["rectangle"]:
                    if (drp["start_point"] and drp["end_point"]) != ():
                        cv2.rectangle(
                            frame,
                            drp["start_point"],
                            drp["end_point"],
                            drp["color"],
                            drp["thickness"],
                        )

            if "text" in drawing_response and drawing_response["text"] is not None:
                for dtp in drawing_response["text"]:
                    if dtp["text"] != "":
                        cv2.putText(
                            frame,
                            dtp["text"],
                            dtp["coord"],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, dtp["color"],
                            dtp["thickness"],
                        )

            if "dot" in drawing_response and drawing_response["dot"] is not None:
                for ddp in drawing_response["dot"]:
                    if ddp["coord"] != ():
                        cv2.circle(
                            frame,
                            ddp["coord"],
                            ddp["radius"],
                            ddp["color"],
                            ddp["thickness"],
                        )

        return frame

    def resized_frame(self, frame, resize_config):
        new_width = int(resize_config.split("x")[0])
        new_height = int(resize_config.split("x")[1])

        try:
            frame = cv2.resize(frame, (new_width, new_height))

        except NameError:
            return None

        return frame
