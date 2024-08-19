import os
from typing import List, Dict
from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from config.config import PipelineConfiguration, get_pipeline_config
from logging import Logger
from logging_utils.log_helper import get_logger

logger: Logger = get_logger(__name__)
APP_CONFIG: PipelineConfiguration = get_pipeline_config()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables.")
    raise OSError("OPENAI_API_KEY is required but not set in the environment.")


class GenerateEmbeddings:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=openai_api_key)
        self.embed_model = APP_CONFIG.openai_embedding_model
        self.langchain_openai_embedding = OpenAIEmbeddings(
            model=self.embed_model,
            api_key=openai_api_key,
        )

    def generate_embeddings_from_dict(self, chunked_articles: List[Dict]) -> List[Dict]:
        """
        Generates embeddings for the content chunks using OpenAI's embedding model and creates a new dictionary.
        """
        embedded_articles = []

        for article in chunked_articles:
            content_chunk = article["content"]

            # Generate embedding for the content chunk
            logger.debug(
                f"Generating embedding for chunk with ID {article['id']} from title '{article['metadata']['title']}'"
            )
            response = (
                self.client.embeddings.create(
                    input=[content_chunk], model=self.embed_model
                )
                .data[0]
                .embedding
            )

            embedded_article = {
                "id": article["id"],
                "values": response,
                "metadata": {"text": content_chunk, **article["metadata"]},
            }

            embedded_articles.append(embedded_article)

        return embedded_articles

    def langchain_generate_embeddings_from_document(
        self, chunks: List[Document]
    ) -> List[List[float]]:
        """
        Generates embeddings for the content chunks using LangChain's embedding model.

        :param chunks: List of Document objects representing content chunks.
        :return: List of embeddings for each chunk.
        :raises: RuntimeError if embedding generation fails.
        """
        if not chunks:
            logger.warning("No document chunks provided for embedding generation.")
            return []

        embeddings = []

        try:
            for chunk in chunks:
                embedding = self.langchain_openai_embedding.embed_documents(
                    chunk.page_content
                )
                embeddings.append(embedding)

            logger.info(f"Generated embeddings for {len(chunks)} document chunks.")
        except Exception as e:
            logger.exception(f"Error generating embeddings with LangChain: {e}")
            raise RuntimeError("Failed to generate embeddings using LangChain.") from e

        return embeddings

    def langchain_embedding_model(self) -> OpenAIEmbeddings:
        """
        Returns the LangChain OpenAIEmbeddings instance.

        :return: OpenAIEmbeddings instance.
        """
        return self.langchain_openai_embedding
