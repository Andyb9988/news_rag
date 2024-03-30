import json
import os

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
            print(f"Error: Configuration file '{self.config_path}' not found.")
            # Handle the case where the file is missing (optional)
            exit(1)
    
    def save_df_to_csv(self, dataframe):
        """
        Save a Pandas DataFrame to a CSV file.

        Parameters:
        dataframe (pandas.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the CSV file will be saved.

        Returns:
        None
        """
        folder_name = 'youtube_dataframe'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
     # Save DataFrame to CSV
        dataframe.to_csv(f"{folder_name}/output.csv", index=False)
        print(f"DataFrame successfully saved to {folder_name}/output.csv")



