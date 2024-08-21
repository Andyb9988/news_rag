import streamlit as st

st.set_page_config(
    page_title="News Summarisation Webpage",
    page_icon="😀",
)


def main():
    """
    The main function for the Generative AI News Summariser application.

    This function sets up the Streamlit web application interface. It includes the title and introductory
    text that guides users on how to use the news summarization feature. Users are instructed to search for
    news on a specific topic, which will then be processed and stored in a vector database. After the news
    articles are loaded, the user can interact with a chatbot to generate summaries based on their queries.

    The function includes:
    - A title for the app.
    - Instructions on how to search for news.
    - Guidance on using the chatbot to summarise the news.
    """
    st.title("Generative AI News Summariser📰")

    st.write("### Welcome to the latest and greatest Generative AI application!")
    st.write(
        "To use it, simply type in a topic in the 'Search the News' tab and then wait until the news is uploaded into the vector database."
    )
    st.write(
        "After this, you can have our chatbot summarise the news for you by prompting what you want to find out."
    )


if __name__ == "__main__":
    main()
