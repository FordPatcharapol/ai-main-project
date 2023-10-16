import cv2
import glob
import time
import pytz
import os

from datetime import datetime
from helpers.api import ApiHelper

api_helpers = ApiHelper()


class VideoRecorder:
    def __init__(self):
        self.time_delete = 1  # in day
        self.frame_rate = 5
        self.time_range = 5  # in seconds
        self.time_ratio = 3600
        self.total_frame_cap = self.frame_rate * self.time_range
        self.frame_counter = 0
        self.video_width = None
        self.video_height = None
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.last_time = str(datetime.fromtimestamp(
            time.time(), tz=pytz.timezone('Asia/Bangkok')).strftime("%y_%m_%d-%H_%M_%S"))
        self.file_name = './videos_capture/' + str(self.last_time) + ".mp4"
        self.out = None
        self.path = "./videos_capture/"
        self.first_frame = True

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def delete_old_videos(self):
        current_time = str(datetime.fromtimestamp(
            time.time(), tz=pytz.timezone('Asia/Bangkok')).strftime("%y_%m_%d-%H_%M_%S"))
        current_timestamp = time.mktime(datetime.strptime(
            current_time, "%y_%m_%d-%H_%M_%S").timetuple())
        video_files = glob.glob(self.path + "*.mp4")

        if len(video_files) != 0:
            video_files.pop()

        for file_path in video_files:
            file_time_str = os.path.basename(file_path).split(".")[0]
            file_timestamp = time.mktime(datetime.strptime(
                str(file_time_str), "%y_%m_%d-%H_%M_%S").timetuple())
            dif_time = (current_timestamp - file_timestamp) / \
                self.time_ratio  # in hours

            if dif_time >= self.time_delete:
                os.remove(file_path)

    def recode_video(self, frame, uuid):

        if self.first_frame:
            self.first_frame = False
            self.frame_counter = 0
            self.last_time = str(datetime.fromtimestamp(
                time.time(), tz=pytz.timezone('Asia/Bangkok')).strftime("%y_%m_%d-%H_%M_%S"))
            self.file_name = './videos_capture/' + str(self.last_time) + ".mp4"
            self.out = cv2.VideoWriter(
                self.file_name, self.fourcc, self.frame_rate, (self.video_width, self.video_height))

        if self.frame_counter > self.total_frame_cap and not self.first_frame:
            self.out.release()

            # send vdo to server
            api_helpers.upload_video_backup(self.file_name, uuid)

            self.frame_counter = 0
            self.last_time = str(datetime.fromtimestamp(
                time.time(), tz=pytz.timezone('Asia/Bangkok')).strftime("%y_%m_%d-%H_%M_%S"))
            self.file_name = './videos_capture/' + str(self.last_time) + ".mp4"
            self.out = cv2.VideoWriter(
                self.file_name, self.fourcc, self.frame_rate, (self.video_width, self.video_height))

        if self.out is not None:
            self.out.write(frame)
            self.frame_counter += 1
