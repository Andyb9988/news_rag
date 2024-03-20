import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json
import tiktoken
from utils import Helper
config_path = "config.json"

class YouTubeSearcher:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def get_video_list_by_search(self, max_results=15):
        """Searches for videos on YouTube using the Youtube Data API.

        Args:
            max_results (int, optional): The maximum number of results to return. Defaults to 15.

        Returns:
            list: A list of video URLs.
        """
        query = str(input("What topic would you like summarised with Youtube videos: "))
        request = self.youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=max_results
        )
        response = request.execute()

        video_urls = []
        for item in response['items']:
            video_id = item['id']['videoId']
            video_url = f'https://www.youtube.com/watch?v={video_id}'
            video_urls.append(video_url)

        return video_urls

    def get_video_metadata(self, video_urls):
        """
        Fetches metadata for a list of YouTube videos.

        This function iterates over a list of YouTube video URLs, extracts the video ID from each URL,
        and uses the YouTube Data API to retrieve the video's metadata.
        The metadata is then compiled into a DataFrame for easy manipulation.

        Parameters:
        - video_urls (list): A list of YouTube video URLs.

        Returns:
        - pandas.DataFrame: A DataFrame containing the metadata for each video.
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
        
        video_meta_df = pd.DataFrame(metadata_list)
        return video_meta_df

    def save_youtube_transcripts(self, video_urls):
        """
            Saves transcripts of YouTube videos to text files.

            This function iterates over a list of YouTube video URLs, extracts the video ID from each URL,
            and uses the YouTube Transcript API to fetch the video's transcript. The transcript is then
            saved to a text file in a specified folder. If a transcript already exists for a video, the function
            skips that video.

            Parameters:
            - urls (list): A list of YouTube video URLs.

            Returns:
            - None
        """
        folder_name = 'youtube_transcripts'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for url in video_urls:
            video_id = url.split('v=')[-1].split('&')[0]

            transcript_file_path = os.path.join(folder_name, f'{video_id}.txt')
            if os.path.exists(transcript_file_path):
                print(f"Transcript already exists for video {video_id}")
                continue
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                # Write transcript to a file
                with open(f'{folder_name}/{video_id}.txt', 'w', encoding='utf-8') as file:
                    for sentence in transcript:
                        file.write(sentence['text'] + '\n')
                print(f"Transcript saved for video {video_id}")
            except Exception as e:
                print(f"Error fetching transcript for {url}: {e}")

class TranscriptProcessor:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    @staticmethod
    def lambda_clean_transcript_text(text):
        text = text.replace("[Music]", "")
        text = text.replace("\n", " ")
        text = text.replace("[Applause]", "")
        return text
    
    def create_transcript_column(self, transcripts_folder: str):
    # Loop through transcript files in the folder
        for filename in os.listdir(transcripts_folder):
            if filename.endswith('.txt'):
                video_id = os.path.splitext(filename)[0]  # Extract video ID from filename
                file_path = os.path.join(transcripts_folder, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    transcript_content = file.read()
            # Find the corresponding row in the DataFrame and add the transcript content
                    self.df.loc[self.df['Video_id'] == video_id, 'transcript'] = transcript_content

        return self.df

    # Apply the function to the "text" column using lambda function
    def clean_transcript_text(self):
        self.df["clean_transcript"] = self.df["transcript"].apply(lambda x: self.lambda_clean_transcript_text(x))
        self.df.drop(columns=["transcript"], inplace=True)
        return self.df


    def count_number_of_tokens(self):
        token_counts = []
        for transcript in self.df['clean_transcript']:
            num_tokens = self.num_tokens_from_string(transcript, "cl100k_base")
            token_counts.append(num_tokens)
        self.df['num_tokens'] = token_counts
        return self.df

    def drop_columns_exceeding_token_length(self):
        threshold_tokens = 8192
        self.df = self.df[self.df['num_tokens'] <= threshold_tokens].reset_index(drop=True)
        print(f"The df has {self.df.shape[0]} rows")
        return self.df

def main():
    # Example usage
    helper = Helper(config_path)
    config = helper.load_config()
    youtube_api_key = config["YOUTUBE_API_KEY"]
    youtube_searcher = YouTubeSearcher(youtube_api_key)

    video_urls = youtube_searcher.get_video_list_by_search(max_results=15)
    df = youtube_searcher.get_video_metadata(video_urls)

    youtube_searcher.save_youtube_transcripts(video_urls)
    # Initialize TranscriptProcessor instance with your DataFrame
    transcript_processor = TranscriptProcessor(df)

    # Call each method sequentially
    transcript_processor.create_transcript_column('youtube_transcripts')
    transcript_processor.clean_transcript_text()
    transcript_processor.count_number_of_tokens()
    transcript_processor.drop_columns_exceeding_token_length()
    helper.save_df_to_csv(df)

if __name__ == '__main__':
    main()