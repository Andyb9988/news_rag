import os
from newsapi import NewsApiClient
from typing import List, Dict

from logging import Logger
from logging_utils.log_helper import get_logger

logger: Logger = get_logger(__name__)

news_api_key = os.getenv("NEWS_API_KEY")


class GetNews:
    def __init__(self) -> None:
        self.newsapi_client = NewsApiClient(api_key=news_api_key)

    def get_content(self, user_input: str, page: int = 1) -> List[Dict]:
        """
        Fetches articles from News API and returns a list of dictionaries containing content and metadata.
        """
        logger.info(f"Fetching articles for page {page}...")
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
        if all_articles["status"] == "ok":

            news_list = []
            for article in all_articles["articles"]:
                logger.info(f"URL from GETNEWS: {article['url']}")
                news_list.append(article["url"])
            return news_list
        else:
            raise ValueError(
                f"API request failed with status: {all_articles['status']}"
            )
