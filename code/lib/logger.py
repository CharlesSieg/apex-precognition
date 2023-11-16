import logging
import os
import signal
import sys

from datetime import datetime


LINE_BREAK = "#######################################################################"
# "%(asctime)s [%(threadName)-10.10s] [%(levelname)-5.5s]  %(message)s"
LOG_FORMAT = "%(asctime)s [%(levelname)-5.5s] %(message)s"


class GracefulExiter:
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("exit flag set to True (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state


class LogManager(object):
    def __init__(self):
        self.logger_name = "stock-prediction"
        self.log = logging.getLogger(self.logger_name)

    def configure(self, log_level, output_destination, path):
        log.setLevel(log_level)

        # Required for writing the log file.
        os.makedirs(f"{output_destination}/{path}", exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%m%d%Y-%H%M%S")
        log_filename = f"{output_destination}/{path}/{timestamp}.log"
        handler = logging.FileHandler(log_filename)

        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        log.addHandler(handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        log.addHandler(handler)


flag = GracefulExiter()

log_manager = LogManager()
log = log_manager.log
log_manager.configure("DEBUG", "./.logs", "foo")
log.info(LINE_BREAK)
