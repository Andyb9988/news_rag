import pandas as pd
from typing import List, Dict, Union, Tuple
import json
import tiktoken
from utils.helper import Helper
from google.cloud import storage
import json
from io import BytesIO
config_path = "utils/config.json"
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TranscriptProcessor:
    def __init__(self):
        self.bucket_name = "youtube-fpl_data"        
    
    def read_transcript_to_df(self) -> pd.DataFrame:
        # Initialize a storage client
        storage_client = storage.Client()

        # Get the bucket object
        bucket = storage_client.get_bucket(self.bucket_name)

        # List objects in the specified bucket and folder
        blobs = bucket.list_blobs(prefix='transcripts/')
        rows = [{'video_id': blob.name.split('/')[-1], 'transcripts': blob.download_as_bytes().decode('utf-8')} for blob in blobs]
        df = pd.DataFrame(rows)
        return df
    
    # Apply the function to the "text" column using lambda function
    def clean_transcript_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df["clean_transcript"] = df["transcripts"].apply(lambda_clean_transcript_text)
        df.drop(columns=["transcripts"], inplace=True)
        return df

    def count_number_of_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        df['num_tokens'] = [num_tokens_from_string(transcript, "cl100k_base") for transcript in df['clean_transcript']]
        # Assign token_counts to df['num_tokens'] after the loop
        return df

    def filter_rows_by_token_length(self, df: pd.DataFrame, threshold_tokens: int = 8192) -> pd.DataFrame:
        df = df.loc[df['num_tokens'] <= threshold_tokens].reset_index(drop=True)
        logging.info(f"The df now has {df.shape[0]} rows")
        return df
    
    def upload_clean_transcript(self, df: pd.DataFrame) -> None:
        # Initialize a storage client
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)

        # Folder in the bucket where you want to upload the transcripts
        folder_name = 'clean_transcripts'

        # Iterate over the DataFrame rows
        for _, row in df.iterrows():
            video_id = row['video_id']
            clean_transcript = row['clean_transcript']
            
            # Create a blob name using the video_id
            blob_name = f'{folder_name}/{video_id}.txt'
            
            # Create a new blob and upload the transcript content
            blob = bucket.blob(blob_name)
            if blob.exists():
                logging.info(f"Skipping {blob_name} as it already exists.")
                continue      
            blob.upload_from_string(clean_transcript, 'text/plain')

        logging.info("The upload of clean transcripts into their new folder is complete.")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
def lambda_clean_transcript_text(text: str) -> str:
    text = text.replace("[Music]", "").replace("\n", " ").replace("[Applause]", "")
    return text



def main():
    # Initialize TranscriptProcessor instance with your DataFrame
    transcript_processor = TranscriptProcessor()
    df  = transcript_processor.read_transcript_to_df()

    # Call each method sequentially
    df = transcript_processor.clean_transcript_text(df)
    df = transcript_processor.count_number_of_tokens(df)
    df = transcript_processor.filter_rows_by_token_length(df)
    transcript_processor.upload_clean_transcript(df)

if __name__ == '__main__':
    main()