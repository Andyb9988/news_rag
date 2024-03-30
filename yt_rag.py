import pandas as pd
import os
from openai import OpenAI
import math
import regex as re
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import openai 
import langchain
import json
from utils import Helper
from google.cloud import storage
config_path = "config.json"

df = pd.read_csv(r"youtube_dataframe/output.csv")

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
    
    def get_embedding(self, bucket_name, folder_path, model= "text-embedding-ada-002"):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        embeddings = []

        for blob in bucket.list_blobs(prefix=folder_path):
            if blob.name.endswith(".txt"):
                with blob.open("r") as f:
                    transcript = f.read() 
                text_values = [text for text in transcript]

                for text in text_values:
                    response = self.client.embeddings.create(input = [text], model=model).data[0].embedding
                    embeddings.append(response)
                sentence_embeddings = np.array(embeddings)
        print(sentence_embeddings.shape)        
        return sentence_embeddings




def main():
    helper = Helper(config_path)
    config = helper.load_config()
  #  pinecone_api = config["PINECONE_API_KEY"]
    openai_api = config["OPENAI_API_KEY"]

   # pinecone = Pinecone(pinecone_api)
    openai_embedding = OpenAI_Embedding(openai_api)

    bucket_name = "youtube-fpl_data"
    folder_path = "clean_transcripts"
    sentence_embeddings = openai_embedding.get_embedding(bucket_name, folder_path)

  #  pinecone.create_pinecone_index()
   

if __name__ == "__main__":
    main()