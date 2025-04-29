import datetime
import os

class Logger:
    _instance = None

    def __new__(cls, logfile=None, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
            if not logfile:
                logfile = datetime.datetime.now().strftime("logfile_%Y%m%d_%H%M%S.txt")
            cls._instance._initialize_logger(logfile)
        return cls._instance

    def _initialize_logger(self, logfile):
        directory = "logs"
        file_path = os.path.join(directory, logfile)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.file_path = file_path

    def log(self, type, text):
        pass