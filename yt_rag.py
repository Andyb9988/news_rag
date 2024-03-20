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

config_path = "config.json"
try:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
        hf_token = config["HF_TOKEN"]
        pinecone_api = config["PINECONE_API_KEY"]
        openai_api = config["OPENAI_API_KEY"]

except FileNotFoundError:
    print(f"Error: Configuration file '{config_path}' not found.")
    # Handle the case where the file is missing (optional)
    exit(1)

df = pd.read_csv(r"youtube_dataframe/output.csv")

pc = Pinecone(
        api_key=pinecone_api
    )
def create_pinecone_index():
    index_name = 'fpl-rag'
    # Now do stuff
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
    index = pc.Index(index_name)
    return index



def main():
    create_pinecone_index()
   

if __name__ == "__main__":
    main()