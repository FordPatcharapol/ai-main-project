import time
import os
import json
from datetime import datetime, timezone
import requests

from helpers.config import load_config, get_api_config
from helpers.logger import get_logger

# logger
logging = get_logger(__name__)

os.environ["TZ"] = "UTC"


class ApiHelper:
    def __init__(self):
        source_config, server_config, using_analytics, using_general, _ = load_config()
        self.source = source_config
        self.server = server_config
        self.user = using_analytics
        self.general = using_general
        self.server_helpers = None
        self.api_request = []
        self.api_last_call = int(time.time())
        self.healthcheck_api_last_call = int(time.time())
        self.api_config = get_api_config()
        self.api_last_retry = int(time.time())
        self.connection_status = True
        self.health_checker_last_retry = int(time.time())
        self.health_checker_status = True
        self.people_data = None

    def get_default_api_struct(self):
        return {
            "timestamp":  '2023-08-09 13:12:55.257806+00',
            "raw_data": [
                {
                    "position": {
                        "0": [
                            20,
                            10
                        ],
                        "1":[
                            30,
                            20
                        ],
                    },
                    "class":"lpr-ocr",
                    "value":{
                            "province": "นครปฐม",
                            "number": "1234"
                    },
                    "text": "license plate",
                }
            ]
        }

    def get_payload_struct(self, class_name, position, text, value):
        return {
            "position": position,
            "class": class_name,
            "value": value,
            "text": text,
        }

    def add_api_struct(self, payloads):
        if self.api_config['enable'] is not True or payloads is None or payloads == []:
            return

        record = self.get_default_api_struct()
        record["timestamp"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f%z")

        record["raw_data"] = payloads
        self.api_request.append(record)

        logging.info("Payload added, Payload size: %s", len(self.api_request))

    def api_call(self):
        if self.api_config['enable'] is not True:
            self.api_request = []
            return

        api_endpoint = self.api_config['server']['address']
        max_buffer = self.api_config['server']['max_frame_buffer']
        period_interval = self.api_config['server']['period_interval']
        retry_connect = self.api_config['server']['retry_connect']

        url = api_endpoint+'/camera_datas/'+self.general['uuid']

        current_time = int(time.time())
        time_diff = current_time - self.api_last_call

        if current_time - self.api_last_retry > retry_connect:
            self.connection_status = True

        if (len(self.api_request) > max_buffer) or time_diff > period_interval:
            if self.connection_status:
                try:
                    if len(self.api_request) > 0:
                        logging.info("API Call, Send analysis result to API....., Payload size: %s", len(
                            self.api_request))
                        logging.debug(json.dumps(self.api_request, indent=4))
                        # Send data to API
                        requests.post(url=url,
                                      json=self.api_request, timeout=3)
                        self.api_request = []

                except requests.Timeout:
                    logging.warning(
                        "API Call, Connection Timeout...., Retry Next Time")

                except requests.ConnectionError:
                    logging.error(
                        "API Call, !! Can't Create Connection to API !!")

                self.api_last_retry = current_time
                self.connection_status = False
                self.api_last_call = int(current_time)

    def healthcheck_call(self, source, analytics, post_process):
        if self.api_config['enable'] is not True:
            return

        period_interval = self.api_config['server']['healthcheck_interval']
        current_time = int(time.time())

        if current_time - self.healthcheck_api_last_call < period_interval:
            return

        status = {
            'post_process': {},
            'analytic_model': {}
        }

        for (_, analytic_key) in enumerate(analytics):
            analytic_val = analytics[analytic_key]
            status['analytic_model'][analytic_key] = {
                "frame_skip": analytic_val['frame_skip'],
                "status":  analytic_val['status'],
            }
        for (_, post_process_key) in enumerate(post_process):
            post_process_val = post_process[post_process_key]
            status['post_process'][post_process_key] = {
                "model_enable": post_process_val['model_enable'],
                "status":  post_process_val['status'],
            }

        healthcheck_api_request = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        }

        api_endpoint = self.api_config['server']['address']
        url = api_endpoint+'/camera/'+self.general['uuid']+'/status'

        retry_connect = self.api_config['server']['retry_connect']
        current_time = int(time.time())

        if current_time - self.health_checker_last_retry > retry_connect:
            self.health_checker_status = True

        if self.health_checker_status:
            try:
                logging.debug(json.dumps(healthcheck_api_request, indent=4))

                # Send data to API
                requests.post(url=url, json=healthcheck_api_request, timeout=3)

                self.healthcheck_api_last_call = int(current_time)

            except requests.Timeout:
                logging.warning(
                    "Health API Call, Connection Timeout...., Retry Next Time")

            except requests.ConnectionError:
                logging.error(
                    "Health API Call, !! Can't Create Connection to API !!")

            self.health_checker_last_retry = current_time
            self.health_checker_status = False

    def get_people_info(self):
        api_endpoint = self.api_config['server']['address']
        url = api_endpoint + '/peoples'

        try:
            logging.info("Load People Info From API...")

            # Get data from API
            self.people_data = requests.get(url=url, timeout=3)
            return self.people_data.json()

        except requests.Timeout:
            logging.warning(
                "Load People Info From API Connection Timeout...., Retry Next Time")

        except requests.ConnectionError:
            logging.error(
                "Load People Info From API, Can't Create Connection to API !!")

        return []

    def upload_video_backup(self, filename, uuid):
        api_endpoint = self.api_config['server']['address']
        url = api_endpoint + '/video/backup'

        with open(filename, 'rb') as video_file:
            video_binary = video_file.read()

        payload = {
            'uuid': uuid,
            'file': (filename, video_binary)
        }
        headers = {}

        try:
            logging.info("Upload Video, Send video capture to API.....")
            requests.post(url=url, files=payload, headers=headers)

        except requests.Timeout:
            logging.warning(
                "Upload Video, Connection Timeout...., Retry Next Time")

        except requests.ConnectionError:
            logging.error(
                "Upload Video, !! Can't Create Connection to API !!")
