import os
import time
import cv2
import pydash

from helpers.api import ApiHelper
from helpers.logger import get_logger
from helpers.config import load_config, enable_all_status, get_enable_key_from_api
from helpers.video import VideoUitls
from helpers.library import LibraryLoader
from helpers.stream.client import ClientHelpers
from helpers.record import VideoRecorder

from ml_postprocess.parking_cal.slot import ClassSlotContainer
from ml_postprocess.object_entry.entry import ClassEntranceContainer

os.environ['TZ'] = 'Asia/Bangkok'

# logger
logging = get_logger(__name__)

library_loader = LibraryLoader()
video_helpers = VideoUitls()
api_helpers = ApiHelper()
slot_container = ClassSlotContainer()
entrance_container = ClassEntranceContainer()
record_helper = VideoRecorder()


def load_models(source, analytics, post_processes):
    """Model loader"""
    list_of_models = []
    analytics_enable_all = enable_all_status()

    for analytic in analytics_enable_all:
        new_model = []

        if analytic == 'lpr_ocr':
            model = library_loader.get_lpr_ocr_model()
            new_model = model.model_loader(source_config=source, analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'lpr_dl':
            model = library_loader.get_lpr_dl_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'gender':
            model = library_loader.get_gender_detection_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'parking_space':
            model = library_loader.get_parking_space_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'car_brand':
            model = library_loader.get_car_brand_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'object_counting':
            model = library_loader.get_object_counting_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'car_model':
            model = library_loader.get_car_model_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        elif analytic == 'emotion':
            model = library_loader.get_emotion_model()
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

        elif analytic == 'face_recognition':
            model = library_loader.get_face_recognition_model()
            new_model = model.model_loader(source_config=source,
                                           analytic_key=analytic,
                                           analytic_value=analytics_enable_all[analytic])

        if new_model != []:
            list_of_models.append(new_model)
        else:
            logging.warning("Config key %s doesn't exists", analytic)

    if post_processes['parking_calculation']['status'] is True:
        model = library_loader.get_parking_space_model()

        for i, rois in enumerate(model.rois):
            slot_container.createSlot(i, rois,
                                      post_processes['parking_calculation']['model_enable'],
                                      analytics['lpr_dl']['frame_skip'])

    if post_processes['object_entry']['status'] is True:
        model = library_loader.get_parking_space_model()
        entrance_container.createEntrace(model.rois,
                                         post_processes['object_entry']['model_enable'])

    # logging.info('\tModels: Success to load uses models')

    return list_of_models


def process_frame(frame, model_list, source, analytics, general, post_processes):
    drawing_responses = []
    api_responses = []
    post_process_response = []

    raw_datas = []

    # Processes
    lpr_ocr_model = library_loader.get_lpr_ocr_model()
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

    for model in model_list:
        model_name = pydash.get(model, 'name', '')
        if model_name == "":
            continue

        if analytics[model_name]['status'] is not True:
            continue

        drawing_response = None
        raw_data = []
        api_response = []

        if model_name == 'lpr_dl':
            drawing_response,  api_response, raw_data = lpr_dl_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'lpr_ocr':
            drawing_response = lpr_ocr_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'parking_space':
            drawing_response, api_response, raw_data = parking_space_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'car_brand':
            drawing_response, api_response, raw_data = car_brand_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'gender':
            drawing_response, api_response, raw_data = gender_detection_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'object_counting':
            drawing_response, api_response, raw_data = object_counting_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'car_model':
            drawing_response, api_response, raw_data = car_model_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'emotion':
            drawing_response, api_response, raw_data = emotion_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'multi_cam':
            drawing_response, api_response, raw_data = multi_camera_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'frauding':
            drawing_response, api_response, raw_data = frauding_detection_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        elif model_name == 'face_recognition':
            drawing_response, api_response, raw_data = face_recognition_model.extract_frame(
                model=model, frame=frame, analytics=analytics)

        if drawing_response is not None:
            drawing_responses.append(drawing_response)

        if api_response:
            api_responses += api_response

        if raw_data:
            raw_datas += raw_data

    if post_processes['parking_calculation']['status']:
        if raw_datas:
            post_process_response = slot_container.addResult(raw_datas, frame)

    if post_processes['object_entry']['status']:
        if raw_datas:
            post_process_response = entrance_container.addResult(raw_datas, frame)

    if post_process_response:
        api_responses.append(post_process_response)

    source_frame = frame.copy()
    frame = video_helpers.overlay_frame(frame, drawing_responses)

    video_helpers.set_frame(frame)
    video_helpers.display("video : "+general['uuid'])

    if source['video_capture']['status']:
        if source['video_capture']['analytic']:
            record_helper.recode_video(frame, general['uuid'])
        elif not source['video_capture']['analytic']:
            record_helper.recode_video(source_frame, general['uuid'])

    record_helper.delete_old_videos()

    api_helpers.add_api_struct(api_responses)
    api_helpers.api_call()


def main(model_list, source, server, analytics, general, post_process):
    """Main program"""

    stream_helper = ClientHelpers(server)
    video_helpers.load_input(source, server)
    record_helper.video_width = int(
        video_helpers.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    record_helper.video_height = int(
        video_helpers.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    record_helper.frame_rate = source['frame_rate']
    record_helper.time_range = source['video_capture']['duration']
    record_helper.total_frame_cap = source['frame_rate'] * \
        source['video_capture']['duration']
    record_helper.time_delete = source['video_capture']['interval']

    prev = 0
    frame_rate = source['frame_rate']
    disable_frame_rate = source['disable_frame_rate_limit']

    while True:
        time_elapsed = time.time() - prev

        try:
            success, frame = video_helpers.read_frame()
            frame = video_helpers.resized_frame(frame, source['resize_frame'])
        except NameError:
            logging.warning('Source: Fail to load source or end of frame')

        if (time_elapsed <= 1. / frame_rate) and not disable_frame_rate:
            continue

        prev = time.time()
        # record_helper.recode_video(frame)
        # record_helper.delete_old_videos()

        if not success or frame is None:
            logging.warning('Source: Fail to load source or end of frame')
            break

        if source["type"] != "stream":
            stream_helper.send_stream(frame)

        # logging.debug('Source: Success to load source')
        # logging.debug( Process: Inference...')
        analytics, post_process = get_enable_key_from_api(
            analytics, post_process)

        process_frame(frame, model_list, source,
                      analytics, general, post_process)
        # logging.debug('Process: Success to inference')

        api_helpers.healthcheck_call(source, analytics, post_process)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    record_helper.out.release()


if __name__ == '__main__':
    # Load Config
    source_config, server_config, using_analytics, using_general, using_post_process = load_config()

    # Load Model
    model_lists = load_models(
        source_config, using_analytics, using_post_process)

    # Inference
    main(model_list=model_lists, source=source_config,
         server=server_config, analytics=using_analytics,
         general=using_general, post_process=using_post_process)
