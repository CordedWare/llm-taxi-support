import logging

def get_logger(name = __name__):
    logging.basicConfig(
        format = "%(asctime)s %(levelname)s %(message)s",
        level = logging.INFO
    )
    return logging.getLogger(name)
