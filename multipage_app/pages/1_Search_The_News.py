import streamlit as st
from utils.vector_database import PineconeHelper
from utils.data_processor import process_and_chunk_articles
from utils.get_embeddings import GenerateEmbeddings
from utils.news_api import GetNews
from logging import Logger
from logging_utils.log_helper import get_logger

# youtube_to_gpt\utils\vector_database.py
logger: Logger = get_logger(__name__)
index_name = "all-news"

news_client = GetNews()
embedding_generator = GenerateEmbeddings()
pc_client = PineconeHelper(index_name=index_name)
pc_client.pinecone_index()

# Streamlit app UI
def main():
    page_config = {
        "page_title": "Search BBC News",
        "layout": "centered",
    }
    st.set_page_config(**page_config)
    st.markdown(
        "<h1 style='text-align: center;'> Ask The News Anything </h1>",
        unsafe_allow_html=True,
    )

    # Text input for search query
    input_query = st.text_input(
        "Enter a topic to search on BBC News and Tech Crunch e.g. 'Artificial Intelligence'"
    )

    # Button to trigger the process
    if st.button("Fetch and Process Articles"):
        with st.spinner("Fetching articles..."):
            try:
                # Fetch articles
                articles = []
                page = 1
                max_pages = 5
                while page <= max_pages:
                    articles_page = news_client.get_content_dict(input_query, page=page)
                    articles.extend(articles_page)
                    page += 1
                    if not articles_page:
                        break
                st.success(f"Fetched {len(articles)} articles.")

                # Process and chunk articles
                chunked_articles = process_and_chunk_articles(articles)
                st.success(f"Chunked into {len(chunked_articles)} pieces.")

                # Generate embeddings
                embedded_articles = embedding_generator.generate_embeddings_from_dict(
                    chunked_articles
                )
                st.success("Generated embeddings for articles.")

                # Upsert data into Pinecone
                pc_client.upsert_data(embedded_articles, namespace=input_query)
                st.success("Data upserted into Pinecone successfully.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
