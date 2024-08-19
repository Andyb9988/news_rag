import streamlit as st
from typing import List, Dict, Any
import unicodedata
import re


def _clean_text(text: str) -> str:
    """
    Cleans the text by removing invisible characters, normalizing it,
    and replacing line breaks with spaces.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Normalize the text to remove any unusual characters
    text = unicodedata.normalize("NFKD", text)

    # Replace any line breaks or multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _extract_source(link: str) -> str:
    # Extract the source from the link
    source = link.split("//")[1].split("/")[0].replace("www.", "")
    source = source.split(".")[0]
    return source.capitalize()


def format_docs_metadata(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen_source = set()
    formatted_docs = []
    for doc in docs:
        metadata = doc.metadata
        title = metadata.get("title", "No title available")
        title = _clean_text(title)
        title = title[:150]
        link = metadata.get("link", "#")
        source = _extract_source(link)

        if (link, source) not in seen_source:
            formatted_docs.append({"title": title, "link": link, "source": source})
            seen_source.add((link, source))
    return formatted_docs


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
