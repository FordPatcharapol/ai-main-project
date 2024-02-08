import asyncio
import os
import time
import cv2
import pytz
import pydash
import torch
import sys
import shutil
from threading import Thread, Event
from datetime import datetime

from helpers.api import ApiHelper
from helpers.argv import get_params
from helpers.logger import get_logger
from helpers.config import get_config_from_load, enable_all_status, get_enable_key_from_api, get_jwt_from_api
from helpers.video import VideoUitls
from helpers.library import LibraryLoader
from helpers.stream.client import ClientHelpers
from helpers.record import VideoRecorder

from ml_postprocess.parking_cal.slot import ClassSlotContainer
from ml_postprocess.object_entry.entry import ClassEntranceContainer
from ml_postprocess.area_counting.area_count import ClassAreaContainer
from ml_postprocess.area_emotion.area_emotion import ClassAreaEmotionContainer
from ml_postprocess.cashier_behavior.cashier_behavior import ClassCashierBehaviorContainer

os.environ['TZ'] = 'Asia/Bangkok'
# torch.set_num_threads(2)  # set number of threahs on cpu

# logger
logging = get_logger(__name__)

library_loader = LibraryLoader()
video_helpers = VideoUitls()
video_helpers.get_config_from_loaded()
api_helpers = ApiHelper()
api_helpers.get_config_from_loaded()
slot_container = ClassSlotContainer()
entrance_container = ClassEntranceContainer()
area_container = ClassAreaContainer()
area_emotion_container = ClassAreaEmotionContainer()
cashier_behavior_container = ClassCashierBehaviorContainer()
record_helper = VideoRecorder()

# package parameter
this = sys.modules[__name__]
this.vdo_filename = None

# params
mode_param = get_params("mode")
thread_event = Event()


async def call_api(source, analytics, post_process, api_responses, priority):
    api_helpers.add_api_struct(api_responses, priority)

    Thread(target=api_helpers.api_call).run()

    Thread(target=api_helpers.healthcheck_call, args=(
        source, analytics, post_process,)).run()

    # Thread(target=api_helpers.upload_video_backup,
    #        args=(video_path, general['uuid'])).run()


def call_api_upload(general, video_path):

    # Thread(target=api_helpers.upload_video_backup,
    #        args=(video_path, general['uuid'], path)).run()

    # api_helpers.upload_video_backup(video_path, general['uuid'])

    thread_event.set()
    thread = Thread(target=api_helpers.upload_video_backup,
                    args=(video_path, general['uuid']))
    thread.start()


def record_video(source_frame, result_frame, record_enable, record_analyze, uuid):

    vdo_name = None
    filename = None

    if record_enable and mode_param == "record":
        if record_analyze:
            vdo_name, filename = record_helper.recode_video(result_frame)
        elif not record_analyze:
            vdo_name, filename = record_helper.recode_video(source_frame)

    if filename is not None and this.vdo_filename != filename:
        this.vdo_filename = filename

    if vdo_name is not None:
        vdo_name_split = vdo_name.split('/')
        dst_vdo_name = record_helper.path + vdo_name_split[3]
        shutil.move(vdo_name, dst_vdo_name)

    return vdo_name


def record_video_analytic(source_frame, result_frame, record_enable, record_analyze, path):

    if record_enable and mode_param == "analytic":
        if record_analyze:
            record_helper.recode_video_analytic(result_frame, path)
        elif not record_analyze:
            record_helper.recode_video_analytic(source_frame, path)

    record_helper.delete_old_videos()


def load_models(source, analytics, post_processes, param):
    """Model loader"""
    list_of_models = []
    analytics_enable_all = enable_all_status()
    entrance_container.model_enable_config = post_processes['object_entry']['model_enable']

    for analytic in analytics_enable_all:
        new_model = []

        if analytic == 'lpr_dl':
            model = library_loader.get_lpr_dl_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'car_brand':
            model = library_loader.get_car_brand_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'multi_cam':
            model = library_loader.get_multi_camera_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'frauding':
            model = library_loader.get_frauding_detection_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'car_model':
            model = library_loader.get_car_model_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'object_counting':
            model = library_loader.get_object_counting_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'parking_space':
            model = library_loader.get_parking_space_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'gender':
            model = library_loader.get_gender_detection_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'emotion':
            model = library_loader.get_emotion_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'face_recognition':
            model = library_loader.get_face_recognition_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        if new_model != []:
            list_of_models.append(new_model)
        else:
            logging.warning("Config key %s doesn't exists", analytic)

    model = library_loader.get_parking_space_model()
    for i, rois in enumerate(model.rois):
        slot_container.createSlot(i, rois,
                                  post_processes['parking_calculation']['model_enable'],
                                  analytics['lpr_dl']['frame_skip'], param['parking_calculation'])

        area_container.createSlot(i, rois, param['area_counting'])

    model = library_loader.get_parking_space_model()
    entrance_container.createEntrace(model.rois,
                                     post_processes['object_entry']['model_enable'], param['object_entry'])
    area_emotion_container.createEntrace(model.rois,
                                         post_processes['area_emotion']['model_enable'], param['area_emotion'])
    cashier_behavior_container.createEntrace(model.rois,
                                             post_processes['cashier_behavior']['model_enable'], param['cashier_behavior'])
    # logging.info('\tModels: Success to load uses models')

    return list_of_models


def data_preprocess(frame, model, classes, conf=0.6):
    result = model(source=frame, conf=conf, verbose=False, classes=classes)
    locations = result[0].boxes.xyxy.tolist()
    classes = result[0].boxes.cls.tolist()

    return locations, classes, result


def process_frame(frame, model_list, source, analytics, general, post_processes, video_capture):
    drawing_responses = []
    api_responses = []
    post_process_response = []
    raw_datas = []
    api_responses_priority = []

    # Processes
    # lpr_ocr_model = library_loader.get_lpr_ocr_model()
    lpr_dl_model = library_loader.get_lpr_dl_model()
    parking_space_model = library_loader.get_parking_space_model()
    car_brand_model = library_loader.get_car_brand_model()
    object_counting_model = library_loader.get_object_counting_model()
    gender_detection_model = library_loader.get_gender_detection_model()
    car_model_model = library_loader.get_car_model_model()
    emotion_model = library_loader.get_emotion_model()
    multi_camera_model = library_loader.get_multi_camera_model()
    frauding_detection_model = library_loader.get_frauding_detection_model()
    face_recognition_model = library_loader.get_face_recognition_model()

    model_sequence = [
        "parking_space",
        "object_counting",
        "face_recognition",
        "emotion",
        "frauding",
        "gender",
        "lpr_dl",
        "car_brand",
        "car_model",
        "multi_cam",
    ]

    face_model_depend = [
        'emotion',
        'face_recognition',
        'gender',
    ]

    object_model_depend = [
        'object_counting',
        'parking_space',
        'frauding',
    ]

    list_of_models_sorted = []

    face_model = None
    face_depend_analytic = []

    object_model = None
    object_depend_analytic = []

    for sort_model_name in model_sequence:
        for m in model_list:
            if pydash.get(m, 'name', '') == sort_model_name:
                list_of_models_sorted.append(m)
            if pydash.get(m, 'name', '') == 'face_recognition':
                face_model = m
            if pydash.get(m, 'name', '') == 'object_counting':
                object_model = m

    # filter dependent key for face
    for model_name in face_model_depend:
        if analytics[model_name]['status'] is True:
            face_depend_analytic.append(model_name)

    if any(item in face_depend_analytic for item in face_model_depend):
        face_from_preprocess, _, _ = data_preprocess(
            model=face_model.load(), frame=frame, classes=[0])

    # filter dependent key for object
    for model_name in object_model_depend:
        if analytics[model_name]['status'] is True:
            object_depend_analytic.append(model_name)

    if any(item in object_depend_analytic for item in object_model_depend):
        _, _, object_result = data_preprocess(
            model=object_model.load(), frame=frame, classes=analytics["object_counting"]["class"], conf=0.45)

    is_detected = False
    for model in list_of_models_sorted:
        model_name = pydash.get(model, 'name', '')
        if model_name == "":
            continue

        if analytics[model_name]['status'] is not True:
            continue

        drawing_response = None
        raw_data = []
        api_response = []
        # api_priority = []

        if model_name == 'object_counting':
            drawing_response, api_response, raw_data, is_detected = object_counting_model.extract_frame(
                result=object_result, analytics=analytics, force_frame=is_detected)

        elif model_name == 'parking_space':
            drawing_response, api_response, raw_data, is_detected = parking_space_model.extract_frame(
                result=object_result, analytics=analytics, force_frame=is_detected)

        elif model_name == 'face_recognition':
            drawing_response, api_response, raw_data, is_detected = face_recognition_model.extract_frame(
                faces=face_from_preprocess, frame=frame, analytics=analytics, force_frame=is_detected)

        elif model_name == 'gender':
            drawing_response, api_response, raw_data, is_detected = gender_detection_model.extract_frame(
                model=model, faces=face_from_preprocess, frame=frame, analytics=analytics, force_frame=is_detected)

        elif model_name == 'emotion':
            drawing_response, api_response, raw_data, is_detected = emotion_model.extract_frame(
                model=model, faces=face_from_preprocess, frame=frame, analytics=analytics, force_frame=is_detected)

        elif model_name == 'car_brand':
            drawing_response, api_response, raw_data, is_detected = car_brand_model.extract_frame(
                model=model, frame=frame, analytics=analytics, force_frame=is_detected)

        elif model_name == 'lpr_dl':
            drawing_response,  api_response, raw_data, is_detected = lpr_dl_model.extract_frame(
                model=model, frame=frame, analytics=analytics, force_frame=is_detected)

        elif model_name == 'car_model':
            drawing_response, api_response, raw_data, is_detected = car_model_model.extract_frame(
                model=model, frame=frame, analytics=analytics, force_frame=is_detected)

        elif model_name == 'multi_cam':
            drawing_response, api_response, raw_data, is_detected = multi_camera_model.extract_frame(
                model=model, frame=frame, analytics=analytics, force_frame=is_detected)

        elif model_name == 'frauding':
            drawing_response, api_response, raw_data, is_detected = frauding_detection_model.extract_frame(
                model=model, frame=frame, analytics=analytics, force_frame=is_detected)

        if drawing_response is not None:
            drawing_responses.append(drawing_response)

        if api_response:
            api_responses += api_response

        if raw_data:
            raw_datas += raw_data

    if raw_datas and post_processes['parking_calculation']['status']:
        slot_container.checkModelEnable(
            post_processes['parking_calculation']['model_enable'])
        post_process_response = slot_container.addResult(raw_datas, frame)
        if post_process_response:
            api_responses_priority.append(post_process_response)

    if raw_datas and post_processes['object_entry']['status']:
        entrance_container.checkModelEnable(
            post_processes['object_entry']['model_enable'])
        post_process_response = entrance_container.addResult(
            raw_datas, frame)
        if post_process_response:
            api_responses_priority.append(post_process_response)

    if raw_datas and post_processes['area_counting']['status']:
        post_process_response = area_container.addResult(
            raw_datas, frame)
        if post_process_response:
            api_responses_priority.append(post_process_response)

    if raw_datas and post_processes['area_emotion']['status']:
        area_emotion_container.checkModelEnable(
            post_processes['area_emotion']['model_enable'])
        post_process_response = area_emotion_container.addResult(
            raw_datas, this.vdo_filename)
        if post_process_response:
            api_responses_priority.append(post_process_response)

    if raw_datas and post_processes['cashier_behavior']['status']:
        cashier_behavior_container.checkModelEnable(
            post_processes['cashier_behavior']['model_enable'])
        post_process_response = cashier_behavior_container.addResult(
            raw_datas, frame)
        if post_process_response:
            api_responses_priority.append(post_process_response)

    # if post_process_response:
    #     api_responses_priority.append(post_process_response)
    #     api_responses.append(post_process_response)

    frame = video_helpers.overlay_frame(frame, drawing_responses)

    # video_helpers.set_frame(frame)
    # video_helpers.display("video : "+general['uuid'], source['resize_preview'])

    return api_responses, frame, api_responses_priority


def main(model_list, source, server, analytics, general, post_process, video_capture):
    """Main program"""

    # stream_helper = ClientHelpers(server)
    # video_helpers.load_input(source, server)

    record_helper.video_width = int(source['resize_frame'].split("x")[0])
    record_helper.target_width = int(source['resize_frame'].split("x")[0])
    record_helper.video_height = int(source['resize_frame'].split("x")[1])
    record_helper.target_height = int(source['resize_frame'].split("x")[1])

    record_helper.frame_rate = source['frame_rate'] * video_capture['speed']
    record_helper.time_range = video_capture['duration']
    record_helper.uuid = general['uuid']
    record_helper.time_lapse = video_capture['speed']

    if mode_param == 'record':
        record_mode(server, source, video_capture, general)

    elif mode_param == 'analytic':
        analytic_mode(model_list, source, server, analytics,
                      general, post_process, video_capture)


def remove_file(remove_list, uuid):

    temp_lst = remove_list.copy()

    for remove_name in remove_list:
        remove_path = './videos_capture/' + uuid + '/' + remove_name
        try:
            os.remove(remove_path)
            temp_lst.remove(remove_name)

        except PermissionError:
            logging.info(
                "[mode analytic] can't remove file name %s", remove_path)

    return temp_lst


def analytic_mode(model_list, source, server, analytics, general, post_process, video_capture):

    stream_helper = ClientHelpers(server)
    remove_lst = []

    while True:

        remove_lst = remove_file(remove_lst, str(general['uuid']))

        files_lst = os.listdir('./videos_capture/' +
                               str(general['uuid']) + '/')
        files_lst = list(set(files_lst).difference(remove_lst))
        files_lst.sort()

        if files_lst == []:
            logging.info("Directory is empty. Sleeping for 5 second...")
            time.sleep(5)
            continue

        file_name = files_lst[0]
        remove_lst.append(file_name)

        record_helper.path = './videos_analytic/' + str(general['uuid']) + '/'
        record_helper.init_directory()

        this.vdo_filename = file_name
        input_vdo_path = './videos_capture/' + \
            str(general['uuid']) + '/' + file_name
        output_vdo_path = record_helper.path + file_name

        logging.info("[mode analytic] read file name: %s", input_vdo_path)

        try:
            video_helpers.load_local_input(input_vdo_path)

        except PermissionError:
            logging.info(
                "[mode analytic] can't open local file name %s", input_vdo_path)
            time.sleep(5)
            continue

        prev = 0
        frame_rate = source['frame_rate']
        disable_frame_rate = source['disable_frame_rate_limit']

        if not disable_frame_rate:
            record_helper.frame_rate = source['frame_rate'] * \
                video_capture['speed']
        else:
            record_helper.frame_rate = 30 * video_capture['speed']

        while True:

            time_elapsed = time.time() - prev

            if (time_elapsed <= 1. / frame_rate) and not disable_frame_rate:
                continue

            try:
                success, frame = video_helpers.read_frame()
                frame = video_helpers.resized_frame(
                    frame, source['resize_frame'])
            except NameError:
                logging.warning('Source: Fail to resize frame')

            if not success or frame is None:
                logging.warning(
                    'Source: Fail to load source or end of frame')
                break

            if source["type"] != "stream" and server['stream_to_server_enable'] is True:
                stream_helper.send_stream(frame)
            else:
                analytics, post_process = get_enable_key_from_api(
                    analytics, post_process)

                source_frame = frame.copy()
                api_responses, frame, api_responses_priority = process_frame(frame, model_list, source,
                                                                             analytics, general, post_process, video_capture)

                record_video_analytic(source_frame=source_frame, result_frame=frame,
                                      record_enable=video_capture['status'], record_analyze=video_capture['analytic'], path=output_vdo_path)

                asyncio.run(call_api(source=source, analytics=analytics,
                                     post_process=post_process, api_responses=api_responses, priority=api_responses_priority))

            video_helpers.set_frame(frame)
            video_helpers.display(
                "video : "+general['uuid'], source['resize_preview'])

            prev = time.time()

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        if not record_helper.first_frame:
            video_helpers.video.release()
            record_helper.out.close()
            record_helper.first_frame = True
            call_api_upload(general=general, video_path=output_vdo_path)
            cv2.destroyAllWindows()


def record_mode(server, source, video_capture, general):
    """ record_mode """

    video_helpers.load_input(source, server)

    record_helper.path = './videos_capture/' + str(general['uuid']) + '/'
    record_helper.init_directory()

    while True:
        try:
            success, frame = video_helpers.read_frame()
            frame = video_helpers.resized_frame(frame, source['resize_frame'])
        except NameError:
            logging.warning('Source: Fail to resize frame')

        if not success or frame is None:
            logging.warning('Source: Fail to load source or end of frame')
            break

        record_helper.frame_rate = 30 * video_capture['speed']
        _ = record_video(source_frame=frame, result_frame=frame,
                         record_enable=video_capture['status'], record_analyze=video_capture['analytic'], uuid=str(general['uuid']))

        video_helpers.set_frame(frame)
        video_helpers.display(
            "video : " + general['uuid'], source['resize_preview'])

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    if not record_helper.first_frame:
        record_helper.out.close()


if __name__ == '__main__':
    logging.info("initializing AI engine..................")

    # Load Config
    source_config, server_config, using_analytics, using_general, using_post_process, using_video_capture, using_param, jwt = get_config_from_load()

    # Load Model
    model_lists = load_models(
        source_config, using_analytics, using_post_process, using_param)

    # Inference
    main(model_list=model_lists, source=source_config,
         server=server_config, analytics=using_analytics,
         general=using_general, post_process=using_post_process,
         video_capture=using_video_capture)
