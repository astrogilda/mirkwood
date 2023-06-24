import logging
import logging.config
from typing import Optional

# Configuration dictionary for the logging module
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s Module:%(module)s Line:%(lineno)d %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": """
                    asctime: %(asctime)s
                    filename: %(filename)s
                    funcName: %(funcName)s
                    levelname: %(levelname)s
                    levelno: %(levelno)s
                    lineno: %(lineno)d
                    message: %(message)s
                    module: %(module)s
                    msec: %(msecs)d
                    pathname: %(pathname)s
            """,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "standard": {
            "formatter": "standard",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "json": {
            "formatter": "json",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "standard": {"level": "INFO", "handlers": ["standard"]},
        "json": {"level": "INFO", "handlers": ["json"]},
    },
}


class LoggingUtility:
    """
    A utility class to handle logging. 

    Methods
    -------
    get_logger(name: str, format: str = "standard", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger
        Returns a logger based on the supplied name and format. If the format argument is not passed in, the default standard formatter is used.
    """

    @staticmethod
    def get_logger(name: str, format: str = "standard", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
        """
        Returns logger based on the supplied format.

        Parameters
        ----------
        name : str
            The name of the logger.

        format : str, optional
            The format of the logger to return. Default is "standard".

        level : str, optional
            The level of the logger to return. Default is "INFO".

        log_file : str, optional
            The file to which the logger will write.

        Returns
        -------
        logging.Logger
            The logger with the specified format.
        """
        logging.config.dictConfig(LOGGING_CONFIG)

        logger = logging.getLogger(name)

        if log_file is not None:
            # Clear existing handlers
            logger.handlers = []

            handler = logging.FileHandler(log_file)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(
                LOGGING_CONFIG['formatters'][format]['format']))

            logger.addHandler(handler)

        return logger
