import os
from logging import Logger
from typing import List, Dict
from openai import OpenAI
from logging_utils.log_helper import get_logger

# logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger: Logger = get_logger(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")


class GenerateEmbeddings:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=openai_api_key)

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
                    input=[content_chunk], model="text-embedding-3-small"
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
