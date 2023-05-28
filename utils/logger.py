import logging
import logging.config

# Configuration dictionary for the logging module
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "decorator_formatter": {
            "format": "%(asctime)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple_formatter": {
            "format": "%(asctime)s Module:%(module)s Line:%(lineno)d %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json_formatter": {
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
        "decorator_handler": {
            "formatter": "decorator_formatter",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "simple_handler": {
            "formatter": "simple_formatter",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "json_handler": {
            "formatter": "json_formatter",
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["decorator_handler"],
        },
        "simple": {"level": "INFO", "handlers": ["simple_handler"]},
        "json": {"level": "INFO", "handlers": ["json_handler"]},
    },
}


class LoggingUtility:
    """
    A utility class to handle logging. This class provides static method to get logger of different formats. 

    Methods
    -------
    get_logger(format: str = "simple") -> logging.Logger
        Returns logger based on the supplied format. If the format argument is not passed in, the default simple formatter is used.
    """

    @staticmethod
    def get_logger(format: str = "simple") -> logging.Logger:
        """
        Returns logger based on the supplied format.

        If the format argument is not passed in, the default simple formatter is used.

        Parameters
        ----------
        format : str, optional
            The format of the logger to return. Default is "simple".

        Returns
        -------
        logging.Logger
            The logger with the specified format.

        Examples
        --------
        Using simple formatter:
        >>> logger = LoggingUtility.get_logger("simple")
        >>> logger.info("This is INFO")

        Using json formatter:
        >>> logger = LoggingUtility.get_logger("json")
        >>> logger.info("This is INFO")
        """
        logging.config.dictConfig(LOGGING_CONFIG)
        return logging.getLogger(format)
