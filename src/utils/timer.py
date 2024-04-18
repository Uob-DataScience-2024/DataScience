import time

from loguru import logger


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        self.end_time = time.time()
        return self

    def elapsed_time(self):
        return self.end_time - self.start_time

    def elapsed_time_str(self):
        diff = (self.end_time - self.start_time) * 1000
        if diff < 1000:
            return f"{diff:.4f} ms"
        diff /= 1000
        if diff < 60:
            return f"{diff:.4f} s"
        diff /= 60
        if diff < 60:
            return f"{diff:.4f} m"
        diff /= 60
        return f"{diff:.4f} h"

    def print(self, msg, level="info"):
        if level == "info":
            logger.info(f"{msg}: {self.elapsed_time_str()}")
        elif level == "debug":
            logger.debug(f"\t{msg}: {self.elapsed_time_str()}")
        elif level == "warning":
            logger.warning(f"-{msg}: {self.elapsed_time_str()}")
        elif level == "error":
            logger.error(f"|-{msg}: {self.elapsed_time_str()}")
        else:
            raise ValueError("Invalid log level")
        return self

    def print_elapsed_time(self):
        logger.info("Elapsed time: {:.2f} seconds".format(self.elapsed_time()))

    def print_elapsed_time_and_reset(self):
        self.print_elapsed_time()
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        return self
