import logging
import logging.config
import os
import coloredlogs
import yaml


def get_logger(
    name, path="config/logger_config.yaml", default_level=logging.INFO
) -> logging.Logger:
    """Function that generates a logger from a configuration file
    or defaults to a preset configuration if a config file is not found."""

    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())

            # Ensure log directory exists
            log_dir = os.path.dirname(config["handlers"]["file_handler"]["filename"])
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            logging.config.dictConfig(config)
            coloredlogs.install()
            logger = logging.getLogger(name)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        logger = logging.getLogger(name)

    return logger
