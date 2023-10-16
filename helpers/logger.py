import logging

# logger
logging.basicConfig(
    datefmt='%d/%m/%Y %H:%M:%S',
    format='|%(asctime)s|%(levelname)s|%(name)s|%(message)s', level=logging.INFO)


def get_logger(name):
    return logging.getLogger(name)
