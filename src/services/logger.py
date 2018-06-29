import time
import logging

from services.common import put_time_stamp


class Logger:
    def __init__(self, filename, append=True, debug=False):
        self.filename = filename
        self.level = logging.DEBUG if debug else logging.INFO
        self.st = 0
        logging.basicConfig(filename=filename, filemode='a' if append else 'w', level=self.level)

    def end_timer(self, text=''):
        et = time.time()
        text = text + ' ' if text else ''
        self.info('{}Finished in {:d} minutes {:d} seconds'
                  .format(text, int((et - self.st)/60), int((et - self.st) % 60)))
        self.st = 0

    def set_config(self, filename, append=True):
        self.filename = filename
        logging.basicConfig(filename=filename, filemode='a' if append else 'w', level=self.level)

    def start_timer(self, text):
        self.st = time.time()
        self.info(text)

    @staticmethod
    def debug(text):
        logging.debug(put_time_stamp(text))

    @staticmethod
    def error(text):
        logging.error(put_time_stamp(text))

    @staticmethod
    def info(text):
        print(text)
        logging.info(put_time_stamp(text))

    @staticmethod
    def warning(text):
        logging.warning(put_time_stamp(text))

