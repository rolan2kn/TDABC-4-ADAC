import os.path

from utilities import utils
import inspect
import time


class Register:
    logger = None

    def __init__(self, name, path):
        self.filename = "{0}/{1}".format(path, name)
        self.FORMAT = '{0}:: {1} :~*~>> {2}::{3}\n'

    def get_ascii_time(self):
        return time.strftime("%y.%m.%d__%H.%M.%S")

    def write(self, msg_type, message):
        paths = "/".join(self.filename.split("/")[:-1])
        if not os.path.isdir(paths):
            os.makedirs(paths)
        self.log_file = open(self.filename, "a")
        line = self.FORMAT.format(self.get_ascii_time(), self.getframeInfo(level=3), msg_type, message)
        self.log_file.write(line)
        self.log_file.close()

    def getframeInfo(self, level=2):
        cf = inspect.currentframe()
        while level > 0:
            if cf is not None:
                cf = cf.f_back
            level -= 1

        if cf is not None:
            token = cf.f_code.co_name
            was_found = False
            name = ""
            str_name = str(cf.f_locals)
            space_parts = str_name.split(" ")
            if len(space_parts) > 1:
                point_parts = space_parts[1].split(".")
                if len(point_parts) > 0:
                    class_or_module_name = point_parts[-1]
                    was_found = class_or_module_name is not None and len(class_or_module_name) > 0
                    if was_found:
                        name = "{0}::{1}".format(class_or_module_name, token)
                del point_parts
            del space_parts

            if was_found:
                return name

        return ""

    @staticmethod
    def create(name = None, path = None):
        if Register.logger is None:
            if path is None:
                path = utils.get_module_path()
            if name is None:
                name = "registro.log"
            Register.logger = Register(name, path)

    @staticmethod
    def destroy():
        if Register.logger is not None:
            del Register.logger
            Register.logger = None

    @staticmethod
    def add_debug_message(msg):
        if Register.logger is None:
            Register.create("register.log")
        Register.logger.write("DEBUG", msg)

    @staticmethod
    def add_info_message(msg):
        if Register.logger is None:
            return

        Register.logger.write("INFO", msg)

    @staticmethod
    def add_warning_message(msg):
        if Register.logger is None:
            Register.create("register.log")

        Register.logger.write("WARNING", msg)

    @staticmethod
    def add_error_message(msg):
        if Register.logger is None:
            Register.create("register.log")

        Register.logger.write("ERROR", msg)

    @staticmethod
    def send_mail_report(msg):
        pass