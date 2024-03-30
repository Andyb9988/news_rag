import os
import math
import regex as re
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import openai 
from openai import OpenAI
import langchain
import json
import string
import re
from utils import Helper
from google.cloud import storage
config_path = "config.json"


class Pinecone:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)

    def create_pinecone_index(self):
        index_name = 'fpl-rag'
        # Now do stuff
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        index = self.pc.Index(index_name)
        return index

class OpenAI_Embedding:
    def __init__(self, api_key):
        self.open_ai_api_key = api_key
        self.client = OpenAI()
        self.bucket_name = "youtube-fpl_data"
        self.folder_path = "clean_transcripts"

    def get_text_transcripts(self):
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)

        flattened_transcripts = []
        for blob in bucket.list_blobs(prefix=self.folder_path):
            if blob.name.endswith(".txt"):
                with blob.open("r") as f:
                    transcript = f.read().translate(str.maketrans('', '', string.punctuation))  
                    flattened_transcripts.append(re.sub(r'\s+', ' ', transcript.strip()))

        return flattened_transcripts
    
    def get_embedding(self, transcript_list, model = "text-embedding-ada-002"):
        embeddings = []
        for string in transcript_list:
            response = self.client.embeddings.create(
                input=[string],
                model=model).data[0].embedding
            embeddings.append(response)
        return embeddings
    

def main():
    helper = Helper(config_path)
    config = helper.load_config()
  #  pinecone_api = config["PINECONE_API_KEY"]
    openai_api = config["OPENAI_API_KEY"]

   # pinecone = Pinecone(pinecone_api)
    openai_embedding = OpenAI_Embedding(openai_api)

    transcripts = openai_embedding.get_text_transcripts()
    sentence_embeddings = openai_embedding.get_embedding(transcripts)

  #  pinecone.create_pinecone_index()
   

if __name__ == "__main__":
    main()