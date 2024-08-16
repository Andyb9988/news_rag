from typing import List, Dict, Union, Optional
import time
from pinecone import ServerlessSpec, PodSpec
from pinecone import Pinecone as Pinecone_Client
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from uuid import uuid4

from logging_utils.log_helper import get_logger
from logging import Logger

logger: Logger = get_logger(__name__)

pinecone_api = os.getenv("PINECONE_API")
openai_api_key = os.getenv("OPENAI_API_KEY")


class PineconeHelper:
    def __init__(self, index_name: str):
        """
        Initializes the Pinecone client with the specified API key and index name.

        Args:
            api_key (str): The API key for Pinecone.
            index_name (str): The name of the Pinecone index to be used.
        """
        self.pc = Pinecone_Client(api_key=pinecone_api)
        self.index_name = index_name
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key,
        )

    def pinecone_index(self):
        """
        Configures the Pinecone index based on the instance's index name. Creates the index if it does not exist.

        Returns:
            Pinecone Index object.
        """
        use_serverless = True
        if self.index_name in self.pc.list_indexes().names():
            self.index = self.pc.Index(self.index_name)

        else:
            # create a new index
            if use_serverless:
                spec = ServerlessSpec(cloud="aws", region="us-east-1")
            else:
                # if not using a starter index, you should specify a pod_type too
                spec = PodSpec()

            self.pc.create_index(
                self.index_name,
                dimension=1536,  # dimensionality of text-embedding-ada-002
                metric="cosine",
                spec=spec,
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            self.index = self.pc.Index(self.index_name)

        logger.info(
            f"Index {self.index_name} stats: {self.index.describe_index_stats()}"
        )
        return self.index

    def upsert_data(self, data_file: Union[Dict, List[Dict]], namespace: str) -> None:
        """
        Upserts data into the configured Pinecone index.

        Args:
            data_file (Union[Dict, List[Dict]]): The data to upsert into the index.
        """

        self.index.upsert(data_file, namespace=f"{namespace}")
        logger.info("upserted datafiles correctly.")

    def langchain_upload_documents_to_vdb(
        self, docs: List[Document], namespace: Optional[str] = None
    ):
        pc_vectorstore = PineconeVectorStore(
            pinecone_api_key=pinecone_api,
            index_name=self.index_name,
            embedding=self.embedding,
            namespace=namespace,
        )
        logger.info(f"Initialised vectorestore {pc_vectorstore}.")
        uuids = [str(uuid4()) for _ in range(len(docs))]
        pc_vectorstore.add_documents(documents=docs, ids=uuids)
        logger.info(f"Documents successfully uploaded to pincecone vectorstore.")
