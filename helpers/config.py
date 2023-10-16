import copy
import json
import requests
import sys
import time
import yaml

from pydash import clone_deep, defaults_deep, get

from helpers.argv import get_params
from helpers.logger import get_logger

# logger
logging = get_logger(__name__)

# package parameter
this = sys.modules[__name__]
this.uuid = ''
this.config = {}
this.config_from_local = {}
this.config_from_server = {}
this.api_config = {}
this.refer_config = {}
this.prev_enable_key = []
this.last_get_config = int(time.time())
this.prev_analytics = {}
this.load_config_from_server = False

config_path = "./config.yml"
config_path_param = get_params("config_path")

api_path = "./api.yml"
api_path_param = get_params("api_config_path")

if config_path_param is not None:
    config_path = config_path_param

if api_path_param is not None:
    api_path = api_path_param

try:
    with open(config_path) as f:
        this.config = yaml.load(f, Loader=yaml.FullLoader)
        this.config_from_local = copy.deepcopy(this.config)

        logging.info('Load config from file: %s', config_path)
except BaseException:
    logging.error('Cannot Load config from file: %s', config_path)

try:
    with open(api_path) as f:
        this.api_config = yaml.load(f, Loader=yaml.FullLoader)

        logging.info('Load config from file: %s', api_path)
except BaseException:
    logging.warn('Cannot Load config from file: %s', api_path)

default_config = {
    'general': {
        'uuid': 'eb33eebb-3a63-454d-b7f7-9e8f888cc2fc'
    },
    'source': {
        'type': 'webcam',
        'input': '0',
        'resize_frame': '1920x1080',
        'cam_enable': True,
        'disable_frame_rate_limit': True,
        'frame_rate': 20,
        'video_capture': {
            'analytic': True,
            'status': True,
            'duration': 15,
            'interval': 1
        }
    },
    'server': {
        'stream_to_server_enable': False,
        'stream_to_url': '192.168.1.23',
        'stream_to_port': 8081,

        'server_listen_address': '',
        'server_listen_port': 8081
    },
    'analytic_model': {
        'lpr_ocr': {
            'status': False,
            'ocr_model': 'tesseract',
            'frame_skip': 5
        },
        'lpr_dl': {
            'status': False,
            'ocr_model': 'jaided',
            'detect_model_path': './ml_libraries/lpr_deep_learning/model/detect_LP.pt',
            'frame_skip': 10,
            'detect_conf': 0.6
        },
        'gender': {
            'status': False,
            'frame_skip': 5,
            'face_model_path': './ml_libraries/gender_detection/model/face_detect.pt',
            'age_model_path': {
                'prototxt': './ml_libraries/gender_detection/model/age.prototxt',
                'caffe': './ml_libraries/gender_detection/model/age.caffemodel'
            },
            'gender_model_path': {
                'prototxt': './ml_libraries/gender_detection/model/gender.prototxt',
                'caffe': './ml_libraries/gender_detection/model/gender.caffemodel'
            }
        },
        'parking_space': {
            'status': False,
            'frame_skip': 10,
            'parking_model_path': './ml_libraries/parking_space/yolov8s.pt',
            'class_path': './ml_libraries/parking_space/coco.txt',
            'rois_path': './ml_libraries/parking_space/data/rois_IMG_2632.csv',
            'class_detect': ['car', 'person']
        },
        'car_brand': {
            'status': False,
            'frame_skip': 10,
            'carlogo_model_path': './ml_libraries/car_brand_detection/model_carlogo.pt'
        },
        'object_counting': {
            'status': False,
            'detect_model_path': './ml_libraries/object_counting/model/yolov8s.pt',
            'frame_skip': 3,
            'class': [0]
        },
        'car_model': {
            'status': False,
            'frame_skip': 10,
            'carmodel_model_path': './ml_libraries/car_model_detection/model_carmodel.pt'
        },
        'emotion': {
            'status': False,
            'frame_skip': 5,
            'face_model_path': './ml_libraries/gender_detection/model/face_detect.pt',
            'emotion_model_path':
                './ml_libraries/emotion_detection/model/CNNModel_feraligned+ck_5emo.h5',
            'label2text_path':
                './ml_libraries/emotion_detection/model/label2text_CNNModel_feraligned+ck_5emo.pkl',
        },
        'multi_cam': {
            'status': False,
            'detect_model_path': './ml_libraries/object_counting/model/yolov8s.pt',
            'frame_skip': 1,
            'class': [0]
        },
        'frauding': {
            'status': False,
            'frame_skip': 1,
            'time_range': 15,
            'pose_model_path': './ml_libraries/frauding_detection/model/yolov8s-pose.pt',
            'detect_model_path': './ml_libraries/frauding_detection/model/frauding.pt'
        },
        'face_recognition': {
            'status': False,
            'frame_skip': 5,
            'period_interval': 3,
            'face_model_path': './ml_libraries/gender_detection/model/face_detect.pt'
        }
    },
    'post_process': {
        'parking_calculation': {
            'status': False,
            'model_enable': ['parking_space', 'lpr_dl', 'car_brand', 'car_model', 'face_recognition'],
        },
        'object_entry': {
            'status': False,
            'model_enable': ['parking_space', 'object_counting', 'face_recognition', 'gender', 'emotion'],
        }
    }
}
default_api_config = {
    'enable': False,
    'server': {
        'address': '',
        'config_enable': False,
        'config_interval': 10,
        'max_frame_buffer': 10,
        'period_interval': 5,
        'healthcheck_interval': 60,
    }
}


def get_api_config():
    return bind_config(this.api_config, default_api_config)


def load_config():
    using_source = {}

    ret = bind_config(this.config, default_config)
    config_from_server = get_config_from_server(ret['general']['uuid'])
    this.refer_config = copy.deepcopy(config_from_server)

    using_source, using_server, using_analytics, using_general, using_post_process = reformat_binding(
        ret, config_from_server)

    return using_source, using_server, using_analytics, using_general, using_post_process


def reformat_binding(ret, config_from_server):
    using_analytics = {}

    if config_from_server != {}:
        config_from_server['data']['source']['cam_enable'] = get(
            config_from_server, "data.source.webcam_enable")

        ret['analytic'] = bind_config(
            get(config_from_server, "data.analytic_model"),  ret['analytic'])
        ret['server'] = bind_config(
            get(config_from_server, "data.server"),  ret['server'])
        ret['source'] = bind_config(
            get(config_from_server, "data.source"),  ret['source'])
        ret['post_process'] = bind_config(
            get(config_from_server, "data.post_process"),  ret['post_process'])

    using_general = {
        'uuid': ret['general']['uuid']
    }
    using_source = {
        'type': ret['source']['type'],
        'input': ret['source']['input'],
        'resize_frame': ret['source']['resize_frame'],
        'cam_enable': ret['source']['cam_enable'],
        'disable_frame_rate_limit': ret['source']['disable_frame_rate_limit'],
        'frame_rate': ret['source']['frame_rate'],
        'video_capture': ret['source']['video_capture']
    }

    using_server = {
        'stream_to_server_enable': ret['server']['stream_to_server_enable'],
        'stream_to_url': ret['server']['stream_to_url'],
        'stream_to_port': ret['server']['stream_to_port'],
        'server_listen_address': ret['server']['server_listen_address'],
        'server_listen_port': ret['server']['server_listen_port'],
    }

    using_post_process = {
        'parking_calculation':  ret['post_process']['parking_calculation'],
        'object_entry':  ret['post_process']['object_entry'],
    }

    for _, process_name in enumerate(using_post_process):
        process = using_post_process[process_name]

        if process['status']:
            for _, enable_model_name in enumerate(process['model_enable']):
                ret['analytic'][enable_model_name]['status'] = True

    if not this.load_config_from_server:
        with open(config_path) as f:
            this.config = yaml.load(f, Loader=yaml.FullLoader)

    for analytic in ret['analytic']:
        using_analytics[analytic] = ret['analytic'][analytic]

        if not this.load_config_from_server:
            using_analytics[analytic]['status'] = this.config['analytic'][analytic]['status']

    if using_source['cam_enable'] is None:
        using_source['cam_enable'] = this.config['source']['cam_enable']

    logging.debug(json.dumps(using_analytics, indent=4))

    return using_source, using_server, using_analytics, using_general, using_post_process


def get_config_from_server(uuid):
    config_data = {}
    this.uuid = uuid
    api_config = get_api_config()

    if api_config['enable'] is not True:
        this.load_config_from_server = False
        return {'data': this.config_from_local}

    if api_config['server']['config_enable'] is not True:
        this.load_config_from_server = False
        return {'data': this.config_from_local}

    if this.config_from_server != {}:
        return this.config_from_server

    # period_interval = 10
    api_endpoint = api_config['server']['address']

    url = api_endpoint+'/camera/'+uuid+'/config'

    # Send data to API
    logging.info('Get Config From Server...')

    try:
        config_data = requests.get(url=url, timeout=5)
        logging.info("Response code :%s", config_data.status_code)
        this.config_from_server = config_data.json()
        this.load_config_from_server = True
    except requests.ConnectionError:
        this.load_config_from_server = False
        if this.refer_config == {}:
            logging.info(
                "Get config, !! Can't Create Connection from Server !!")
            logging.info("Load Default Config From Local...")
            this.config_from_server = {"data": this.config_from_local}
        else:
            logging.info("Using Last Config Loaded...")
            this.config_from_server = this.refer_config

    return this.config_from_server


def bind_config(new_config, default_config):
    if new_config is None:
        new_config = {}

    return defaults_deep(new_config, default_config)


def enable_all_status():
    temp_refer = copy.deepcopy(this.refer_config)
    key_name = temp_refer['data']

    if 'analytic_model' in key_name:
        analytics_all = temp_refer['data']['analytic_model']
    elif 'analytic' in key_name:
        analytics_all = temp_refer['data']['analytic']

    for key in analytics_all:
        analytics_all[key]['status'] = True

    return analytics_all


def get_enable_key_from_api(analytics, post_process):
    api_config = get_api_config()
    current_time = int(time.time())
    time_diff = current_time - this.last_get_config

    if time_diff > api_config['server']['config_interval']:
        this.config_from_server = {}
        _, _, using_analytics, _, using_post_process = load_config()
        this.last_get_config = current_time

        return using_analytics, using_post_process

    return analytics, post_process
