import os
import time
from typing import List, Dict, Union, Optional
from uuid import uuid4
from pinecone import Pinecone as Pinecone_Client, ServerlessSpec, PodSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from config.config import PipelineConfiguration, get_pipeline_config
from logging import Logger
from logging_utils.log_helper import get_logger

APP_CONFIG: PipelineConfiguration = get_pipeline_config()
logger: Logger = get_logger(__name__)

pinecone_api = os.getenv("PINECONE_API")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not pinecone_api:
    logger.error("PINECONE_API is not set in the environment variables.")
    raise OSError("PINECONE_API is required but not set in the environment.")
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables.")
    raise OSError("OPENAI_API_KEY is required but not set in the environment.")


class PineconeHelper:
    def __init__(self, index_name: str):
        """
        Initializes the Pinecone client with the specified API key and index name.

        Args:
            api_key (str): The API key for Pinecone.
            index_name (str): The name of the Pinecone index to be used.
        """
        self.pc = Pinecone_Client(api_key=pinecone_api)
        self.embed_model = APP_CONFIG.openai_embedding_model
        self.index_name = index_name
        self.embedding = OpenAIEmbeddings(
            model=self.embed_model,
            api_key=openai_api_key,
        )
        self.index = None
        logger.info(f"PineconeHelper initialized with index '{self.index_name}'.")

    def pinecone_index(self) -> Pinecone_Client:
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

    def upsert_data(
        self, data_file: Union[Dict, List[Dict]], namespace: Optional[str] = None
    ) -> None:
        """
        Upserts data into the configured Pinecone index.

        Args:
            data_file (Union[Dict, List[Dict]]): The data to upsert into the index.
        """

        try:
            self.index.upsert(data_file, namespace=namespace or "")
            logger.info(
                f"Successfully upserted data into index '{self.index_name}' under namespace '{namespace}'."
            )
        except Exception as e:
            logger.exception(
                f"Failed to upsert data into Pinecone index '{self.index_name}': {e}"
            )
            raise RuntimeError("Failed to upsert data into Pinecone index.") from e

    def langchain_upload_documents_to_vdb(
        self, docs: List[Document], namespace: Optional[str] = None
    ):
        try:
            uuids = [str(uuid4()) for _ in range(len(docs))]
            namespace = namespace or ""

            pc_vectorstore = PineconeVectorStore(
                pinecone_api_key=pinecone_api,
                index_name=self.index_name,
                embedding=self.embedding,
                namespace=namespace,
            )

            logger.info(
                f"Initialized vector store with namespace '{namespace}' for index '{self.index_name}'."
            )
            pc_vectorstore.add_documents(documents=docs, ids=uuids)
            logger.info("Documents successfully uploaded to Pinecone vectorstore.")
        except Exception as e:
            logger.exception(f"Failed to upload documents to Pinecone vectorstore: {e}")
            raise RuntimeError(
                "Failed to upload documents to Pinecone vectorstore."
            ) from e

    def langchain_pinecone_vectorstore(
        self, embeddings: OpenAIEmbeddings
    ) -> PineconeVectorStore:
        """
        Initializes and returns a Pinecone vector store using LangChain.

        Args:
            embeddings (OpenAIEmbeddings): The embeddings model to use.

        Returns:
            PineconeVectorStore: The initialized vector store.
        """
        try:
            vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=embeddings,
                pinecone_api_key=pinecone_api,
            )
            logger.info(
                f"Initialized Pinecone vector store for index '{self.index_name}'."
            )
            return vectorstore
        except Exception as e:
            logger.exception(f"Failed to initialize Pinecone vector store: {e}")
            raise RuntimeError("Failed to initialize Pinecone vector store.") from e

    def pinecone_stats(self) -> str:
        """
        Retrieves and returns the statistics for the Pinecone index.

        Returns:
            str: A string representation of the index stats.
        """
        self.index = self.pc.Index(self.index_name)
        index_stats = self.index.describe_index_stats()
        logger.info(f"Retrieved stats for Pinecone index '{self.index_name}'.")
        return f"Index stats: {index_stats}"

    def pinecone_delete_index_by_ids(
        self,
        namespace: Optional[str] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Deletes specific entries from the Pinecone index by their IDs.

        Args:
            namespace (Optional[str]): The namespace within the index.
            ids (Optional[List[str]]): The IDs of the entries to delete.
        """
        vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            pinecone_api_key=pinecone_api,
            namespace=namespace,
            ids=ids,
        )
        vectorstore.delete()
        logger.info(
            f"Deleted IDs from namespace '{namespace}' in index '{self.index_name}'."
        )

    def pinecone_delete_index_by_namespace(
        self, namespace: Optional[str] = None
    ) -> None:
        """
        Deletes all entries within a specific namespace in the Pinecone index.

        Args:
            namespace (Optional[str]): The namespace to delete.
        """

        index = self.pc.Index(name=self.index_name)
        index.delete(delete_all=True, namespace=namespace)
        logger.info(
            f"Deleted all entries in namespace '{namespace}' from index '{self.index_name}'."
        )
