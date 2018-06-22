import time


class Logger:
    def __init__(self, mode=1):
        self.mode = mode
        self.st = 0

    def start_timer(self, text):
        self.st = time.time()
        self.log(text)

    def end_timer(self, text=''):
        et = time.time()
        text = text + ' - ' if text else ''
        self.log('{}Finished in {:d} minutes {:d} seconds'
                 .format(text, int((et - self.st)/60), int((et - self.st) % 60)))
        self.st = 0

    def log(self, text):
        if self.mode == 1:
            print(text)


