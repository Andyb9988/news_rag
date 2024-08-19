import os
from newsapi import NewsApiClient
from typing import List, Dict, Optional

from logging import Logger
from logging_utils.log_helper import get_logger

logger: Logger = get_logger(__name__)

news_api_key = os.getenv("NEWS_API_KEY")
if not news_api_key:
    logger.error("NEWS_API_KEY is not set in the environment variables.")
    raise OSError("NEWS_API_KEY is required but not set in the environment.")


class GetNews:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.newsapi_client = NewsApiClient(api_key=api_key or news_api_key)

    def get_content(self, user_input: str, page: int = 1) -> List[Dict]:
        """
        Fetches articles from News API and returns a list of dictionaries containing content and metadata.
        """
        logger.info(f"Fetching articles for page {page}...")
        try:
            all_articles = self.newsapi_client.get_everything(
                q=f"{user_input}",
                sources="bbc-news,the-verge",
                domains="techcrunch.com, bbc.co.uk",
                from_param="2024-08-07",
                to="2024-08-15",
                language="en",
                sort_by="relevancy",
                page=page,
            )
            logger.info(f"API Response: {all_articles}")
            if all_articles["status"] != "ok":
                logger.error(
                    f"API request failed with status: {all_articles['status']}"
                )
                raise ValueError(
                    f"API request failed with status: {all_articles['status']}"
                )

            articles = all_articles.get("articles", [])
            if not articles:
                logger.warning("No articles were found for the given query.")
                return []

            news_list = [article["url"] for article in articles]
            logger.info(f"Retrieved {len(news_list)} articles.")
            return news_list

        except Exception as e:
            logger.exception(f"An error occurred while fetching news articles: {e}")
            raise RuntimeError(f"Failed to fetch news articles: {e}") from e
