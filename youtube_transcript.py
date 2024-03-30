import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd
from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi
import json
import tiktoken
from utils import Helper
from google.cloud import storage
import json
from io import BytesIO
config_path = "config.json"

class YouTubeSearcher:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.bucket_name = "youtube-fpl_data"
    
    def get_video_list_by_search(self, max_results: int = 5) -> List[str]:
        """Searches for videos on YouTube using the Youtube Data API.

        Args:
            max_results (int, optional): The maximum number of results to return. Defaults to 15.

        Returns:
            list: A list of video URLs.
        """
        video_urls = []

        while True:
            query = str(input("What topic would you like summarised with Youtube videos: "))

            if query.lower() == 'done':
                break

            request = self.youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=max_results
            )
            response = request.execute()
            video_urls = [f'https://www.youtube.com/watch?v={item["id"]["videoId"]}' for item in response['items']]
            
            print(f"Number of items in the list: {len(video_urls)}")

            if len(video_urls) >= 5:
                print("Max video limit (5) reached")
                break

        return video_urls[:max_results]


    def get_video_metadata(self, video_urls: List[str]) -> List[Dict]:
        """
        Fetches metadata for a list of YouTube videos.

        This function iterates over a list of YouTube video URLs, extracts the video ID from each URL,
        and uses the YouTube Data API to retrieve the video's metadata.
        The metadata is then compiled into a DataFrame for easy manipulation.

        Parameters:
        - video_urls (list): A list of YouTube video URLs.

        Returns:
        - List of dictionaries: A list containing the metadata for each video.
        """
        metadata_list = []
        for video_url in video_urls:
            video_id = video_url.split('v=')[-1].split('&')[0]

            request = self.youtube.videos().list(
                part='snippet',
                id=video_id
            )
            response = request.execute()
            video_items = response['items']

            if video_items:
                video_item = video_items[0]
                snippet = video_item['snippet']
                metadata = {
                    'Title': snippet['title'],
                    'Published_at': snippet['publishedAt'],
                    'Channel': snippet["channelTitle"],
                    'URL': f"https://www.youtube.com/watch?v={video_id}",
                    "Video_id": video_id
                }
                metadata_list.append(metadata)
            else:
                print(f'No video found with ID {video_id}')

        return metadata_list

    def upload_transcript_to_gcs(self, video_urls: List[str], folder_name: str):
        """
        Fetches transcripts of YouTube videos and uploads them directly to GCS as .txt files.

        Parameters:
        - video_urls (list): A list of YouTube video URLs.
        - bucket_name (str): The name of the GCS bucket.
        - folder_name (str): The name of the folder within the GCS bucket.
        """
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        
        for url in video_urls:
            video_id = url.split('v=')[-1].split('&')[0]
            
            try:
                blob_path = f"{folder_name}/{video_id}.txt"
                blob = bucket.blob(blob_path)
                
                if not blob.exists():
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    # Open the blob for writing as a text file
                    with blob.open("w", encoding='utf-8') as f:
                        for sentence in transcript:
                            f.write(sentence['text'] + '\n')
                    print(f"Uploaded transcript for video {video_id} to {self.bucket_name}/{blob_path}")
                    
            except Exception as e:
                print(f"Error fetching/uploading transcript for {url}: {e}")

    def upload_dicts_to_gcs(self, folder_name: str, data_list: List[Dict]):
        """
        Uploads a list of dictionaries as individual JSON files to GCS.

        Args:
            bucket_name (str): The name of the GCS bucket.
            folder_name (str): The name of the folder within the bucket.
            data_list (list): A list of dictionaries to upload.
        """    
    # Initialize the GCS client and get the bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        
        # Iterate through each dictionary in the list
        for data in data_list:
            video_id = data.get('Video_id')
            if not video_id:
                print("Dictionary missing 'video_id'. Skipping...")
                continue
            
            # Convert the dictionary to a JSON string
            data_str = json.dumps(data)
            
            # Create a blob (file) in the specified folder, named after the video ID
            blob_name = f"metadata/{video_id}.json"
            blob = bucket.blob(f"{folder_name}/{video_id}.json")
            
            #check if the blob exists
            if blob.exists():
                print(f"File {blob_name} already exists in bucket {self.bucket_name}")
            else:
        # Upload the JSON string to the blob if it does not exist
                blob.upload_from_string(data_str)
                print(f"Uploaded {video_id}.json to {folder_name}/ in bucket {self.bucket_name}")
     
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
        print(f"The df now has {df.shape[0]} rows")
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
            blob_name = f'{folder_name}/{video_id}'
            
            # Create a new blob and upload the transcript content
            blob = bucket.blob(blob_name)
            if blob.exists():
                print(f"Skipping {blob_name} as it already exists.")
                continue      
            blob.upload_from_string(clean_transcript, 'text/plain')

        print("Upload completed.")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
def lambda_clean_transcript_text(text: str) -> str:
    text = text.replace("[Music]", "").replace("\n", " ").replace("[Applause]", "")
    return text

def main():
    # Example usage
    helper = Helper(config_path)
    config = helper.load_config()
    youtube_api_key = config["YOUTUBE_API_KEY"]
    youtube_searcher = YouTubeSearcher(youtube_api_key)
    meta_folder_name = "metadata"
    transcript_folder_name = "transcripts"
    # Search for the youtube URL's
    video_urls = youtube_searcher.get_video_list_by_search(max_results=5)

    # Gets video metadata and uploads metadata and transcript to a gcs bucket
    meta_data_list = youtube_searcher.get_video_metadata(video_urls)
    youtube_searcher.upload_dicts_to_gcs(meta_folder_name, meta_data_list)
    youtube_searcher.upload_transcript_to_gcs(video_urls, transcript_folder_name)

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