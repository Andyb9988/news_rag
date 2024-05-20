import pandas as pd
import logging
from google.cloud import storage
import tiktoken
import logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class TranscriptProcessor:
    def __init__(self, folder_name = "other"):
        """Initializes the TranscriptProcessor with a specific Google Cloud Storage bucket name."""
        self.bucket_name = "youtube-transcript-data"
        self.folder_name = folder_name

    def read_transcript_to_df(self) -> pd.DataFrame:
        """
        Reads transcripts from a specified Google Cloud Storage bucket and returns them as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame where each row contains a video ID and its corresponding transcript.
        """
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=f'{self.folder_name}/{self.folder_name}_transcripts/')
        rows = [{'video_id': blob.name.split('/')[-1], 'transcripts': blob.download_as_bytes().decode('utf-8')} for blob in blobs]
        df = pd.DataFrame(rows)
        logging.info("Transcript read to a pandas dataframe.")
        return df

    def clean_transcript_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the transcript text in the DataFrame by removing specific unwanted strings and returns the modified DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the transcripts.
        
        Returns:
            pd.DataFrame: The DataFrame with cleaned transcripts.
        """
        df["clean_transcript"] = df["transcripts"].apply(self._lambda_clean_transcript_text)
        df.drop(columns=["transcripts"], inplace=True)
        logging.info("Clean transcript column added to the dataset.")
        return df

    def count_number_of_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Counts the number of tokens in each transcript using a specified encoding and adds this count to the DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the cleaned transcripts.
        
        Returns:
            pd.DataFrame: The DataFrame with a new column 'num_tokens' representing the number of tokens in each transcript.
        """
        df['num_tokens'] = [self._num_tokens_from_string(transcript, "cl100k_base") for transcript in df['clean_transcript']]
        return df

    def filter_rows_by_token_length(self, df: pd.DataFrame, threshold_tokens: int = 8192) -> pd.DataFrame:
        """
        Filters rows in the DataFrame based on the number of tokens, removing rows that exceed a specified threshold.
        
        Args:
            df (pd.DataFrame): The DataFrame to filter.
            threshold_tokens (int): The maximum number of tokens allowed per transcript.
        
        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        df = df.loc[df['num_tokens'] <= threshold_tokens].reset_index(drop=True)
        logging.info(f"The df now has {df.shape[0]} rows")
        return df

    def upload_clean_transcript(self, df: pd.DataFrame) -> None:
        """
        Uploads cleaned transcripts back to a specified folder in the Google Cloud Storage bucket.
        
        Args:
            df (pd.DataFrame): The DataFrame containing video IDs and their corresponding cleaned transcripts.
        """
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)
        folder_name = f'{self.folder_name}/{self.folder_name}_clean_transcripts'
        for _, row in df.iterrows():
            video_id = row['video_id']
            clean_transcript = row['clean_transcript']
            blob_name = f'{folder_name}/{video_id}'
            blob = bucket.blob(blob_name)
            if not blob.exists():
                blob.upload_from_string(clean_transcript, 'text/plain')
            else:
                logging.info(f"Skipping {blob_name} as it already exists.")
        logging.info("The upload of clean transcripts into their new folder is complete.")


    def upload_clean_transcript1(self, df: pd.DataFrame) -> None:
        """
        Uploads cleaned transcripts back to a specified folder in the Google Cloud Storage bucket.
        
        Args:
            df (pd.DataFrame): The DataFrame containing video IDs and their corresponding cleaned transcripts.
        """
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)
        folder_name = f'{self.folder_name}/{self.folder_name}_clean_transcripts'

        for _, row in df.iterrows():
            video_id = row['video_id']
            clean_transcript = row['clean_transcript']
            if clean_transcript.strip():  # Check if the transcript is not just whitespace
                blob_name = f'{folder_name}/{video_id}'  # Ensure to include the file extension
                blob = bucket.blob(blob_name)
                if not blob.exists():
                    blob.upload_from_string(clean_transcript, 'text/plain')
                    logging.info(f"Uploaded clean transcript for video {video_id} to {blob_name}.")
                else:
                    logging.info(f"Skipping {blob_name} as it already exists.")
            else:
                logging.info(f"No content to upload for video {video_id}. Skipping.")
        logging.info("The upload of clean transcripts into their new folder is complete.")

    def _num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Calculates the number of tokens in a string using a specified encoding."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def _lambda_clean_transcript_text(self, text: str) -> str:
        """Cleans the transcript text by removing specific markers and whitespace."""
        text = text.replace("[Music]", "").replace("\n", " ").replace("[Applause]", "")
        return text

    def process_transcripts(self):
        """
        Main function to process transcripts:
    - Reads transcripts into a DataFrame.
    - Cleans the transcript text.
    - Counts the number of tokens.
    - Filters the DataFrame based on token count.
    - Uploads the cleaned transcripts.
        """
        df = self.read_transcript_to_df()
        df = self.clean_transcript_text(df)
        df = self.count_number_of_tokens(df)
        df = self.filter_rows_by_token_length(df)
        self.upload_clean_transcript1(df)

def main():
    """
    Main function to process transcripts:
    """
    transcript_processor = TranscriptProcessor(folder_name="golf")
    transcript_processor.process_transcripts()

if __name__ == '__main__':
    main()