import logging
import logging.config


"""Logging configuration"""

_LOGGER_NAME = "dust3r_logger"
_LOGGER_LEVEL = "DEBUG"

config_dict = {
    "version": 1,
    "loggers": {
        _LOGGER_NAME: {"level": _LOGGER_LEVEL, "handlers": ["consoleHandler"]},
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": _LOGGER_LEVEL,
            "formatter": "simpleFormatter",
            "stream": "ext://sys.stdout",
        }
    },
    "formatters": {
        "simpleFormatter": {
            "format": (
                "%(asctime)s - %(levelname)s [%(process)s "
                "- %(filename)s:%(lineno)d] %(message)s"
            )
        }
    },
}

logging.config.dictConfig(config_dict)

dust3r_logger = logging.getLogger(_LOGGER_NAME)
