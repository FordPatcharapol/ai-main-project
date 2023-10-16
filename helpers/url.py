import os
import re
import requests
from helpers.logger import get_logger

# logger
logging = get_logger(__name__)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)

    save_response_content(response, destination)


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def read_path(file_path, model_name, file_name):

    regex = ("((http|https)://)(www.)?" +
             "[a-zA-Z0-9@:%._\\+~#?&//=]" +
             "{2,256}\\.[a-z]" +
             "{2,6}\\b([-a-zA-Z0-9@:%" +
             "._\\+~#?&//=]*)")

    https_pattern = re.compile(regex)

    if not (re.search(https_pattern, file_path)):
        return file_path

    id_pattern = re.compile(r"/([a-zA-Z0-9_-]{33})/")
    match = re.search(id_pattern, file_path)

    if not match:
        return logging.error("Can't download file from this URL (model: %s, file_name: %s)", model_name, file_name)

    file_id = match.group(1)
    dest_path = "./temp_model/" + model_name + "/" + file_name

    if os.path.exists(dest_path):
        return dest_path

    logging.info("Downloading file into %s...", dest_path)
    download_file_from_google_drive(file_id, dest_path)

    return dest_path


def model_path_checker(analytic):
    dir_list = os.listdir('./temp_model/')
    intersection = len(set(analytic).intersection(set(dir_list)))
    union = len(set(analytic).union(set(dir_list)))

    jaccard_similarity = intersection / union
    percentage_similarity = jaccard_similarity * 100

    if percentage_similarity > 50:
        return True
    else:
        return False
