import streamlit as st

st.set_page_config(
    page_title="News Summarisation Webpage",
    page_icon="😀",
)


def main():
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
