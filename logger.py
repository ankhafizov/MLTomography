import logging.handlers
import os
import sys

SCRIPT_PATH = sys.argv[0]
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
LOG_FILE_DIR = os.path.join(SCRIPT_DIR, 'log')
LOG_FILE_NAME = f'{SCRIPT_NAME}.log'
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)


def level_value(level):
    if type(level) is int:
        return level
    if type(level) is str:
        try:
            log_level = getattr(logging, level)
        except AttributeError as error:
            logging.error(error)
            return logging.NOTSET
        else:
            if type(log_level) is int:
                return log_level
            else:
                logging.error(f'module \'logging\' has no attribute \'{level}\'')
                return logging.NOTSET
    return logging.NOTSET


CONSOLE_LOG_LEVEL = level_value(os.getenv('CONSOLE_LOG_LEVEL', default=logging.NOTSET))
FILE_LOG_LEVEL = level_value(os.getenv('FILE_LOG_LEVEL', default=logging.NOTSET))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s')

if CONSOLE_LOG_LEVEL:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_format = log_formatter
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

if FILE_LOG_LEVEL:
    if not os.path.exists(LOG_FILE_DIR):
        os.makedirs(LOG_FILE_DIR)
    file_handler = logging.handlers.RotatingFileHandler(LOG_FILE_PATH, maxBytes=100*1024, backupCount=10)
    file_handler.setLevel(FILE_LOG_LEVEL)
    file_format = log_formatter
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)