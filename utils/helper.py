import json
import os
import logging


class Helper:
    def __init__(self, config_path) -> None:
        self.config_path = config_path

    def load_config(self):
        """Loads configuration from a JSON file.

        Returns:
            dict: The loaded configuration dictionary, or None if there's an error.
        """
        try:
            with open(self.config_path, "r") as config_file:
                config = json.load(config_file)
                return config
        except FileNotFoundError:
            logging.error(f"Error: Configuration file '{self.config_path}' not found.")
            # Handle the case where the file is missing (optional)
            exit(1)
