import time
import logging

from services.common import put_time_stamp


class Logger:
    def __init__(self):
        self.st = 0

    def end_timer(self, text=''):
        t = time.time() - self.st
        h = int(t / 3600)
        t = t - h * 3600
        m = int(t / 60)
        s = int(t % 60)
        text = text + (' ' if text else '') + 'Finished in' + (' {:d} hours'.format(h) if h > 0 else '') \
                    + (' {:d} minutes'.format(m) if m > 0 else '') + ' {:d} seconds.'.format(s)
        self.info(text)
        self.st = 0

    @staticmethod
    def set_config(filename, append=True, debug=False):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(filename=filename, filemode='a' if append else 'w', level=level)

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
