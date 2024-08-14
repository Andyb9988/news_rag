import requests
import os

news_api_key = os.getenv("NEWS_API_KEY")
base_url = "https://newsapi.org/v2/everything"

params = {
    "apiKey": news_api_key,
    "domains": "fantasyfootballscout.co.uk",
    "language": "en",
    "sortBy": "publishedAt",
}


def get_data():
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Print the articles
        for article in data["articles"]:
            print(f"Title: {article['title']}")
            print(f"Published at: {article['publishedAt']}")
            print(f"Description: {article['description']}")
            print("---")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
