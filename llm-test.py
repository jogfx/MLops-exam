# %%
# Install necessary packages
!pip install langchain together sqlalchemy langchain_community langchain_experimental transformers


# %%
!pip install langchain-core

# %%
import os
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain, SQLDatabaseChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableLambda
from pydantic import BaseModel
from typing import List
from huggingface_hub import login
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama

# %%
import sqlite3
import pandas as pd
import pickle

# %%
# Define the SQLite database connection string
sqlite_uri = "sqlite:////Users/patrickribersorensen/Documents/BDS/Coding/LLM_news/news_database.db"
db = SQLDatabase.from_uri(sqlite_uri)

# %%
def setup():
    # Load your trained SVM model
    with open("svm_classifier.pkl", "rb") as f:
        svm_classifier = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Connect to SQLite database
    conn = sqlite3.connect("news_database.db")
    cursor = conn.cursor()

    return svm_classifier, vectorizer, cursor

svm_classifier, vectorizer, cursor = setup()

# %%
# Function to fetch news articles from the database
def fetch_news():
    cursor.execute("SELECT id, title, published_at, name, content, url, url_to_image FROM news_items")
    news_data = cursor.fetchall()
    news_df = pd.DataFrame(news_data, columns=['ID', 'Title', 'Published_At', 'Name', 'Content', 'URL', 'URL_To_Image'])
    return news_df

# Load the dataset of news articles
news_df = fetch_news()

# %%
news_df

# %%
# import pandas as pd
# import sqlite3
# import pickle

# # Define the setup function
# def setup():
#     # Load your trained SVM model
#     with open("svm_classifier.pkl", "rb") as f:
#         svm_classifier = pickle.load(f)

#     # Load the TF-IDF vectorizer
#     with open("tfidf_vectorizer.pkl", "rb") as f:
#         vectorizer = pickle.load(f)

#     # Connect to SQLite database
#     conn = sqlite3.connect("news_database.db")
#     cursor = conn.cursor()

#     return svm_classifier, vectorizer, cursor

# # Call the setup function to get SVM classifier, TF-IDF vectorizer, and cursor
# svm_classifier, vectorizer, cursor = setup()

# # Function to predict sentiment for a given title
# def predict_sentiment(title):
#     # Vectorize the title
#     title_vectorized = vectorizer.transform([title])

#     # Make prediction using SVM classifier
#     prediction = svm_classifier.predict(title_vectorized)

#     # Return the predicted sentiment
#     return prediction[0]

# # Function to fetch the 10 newest articles for each publication and predict sentiment
# def predict_sentiment_for_newest_articles():
#     # Dictionary to store sentiment predictions for each publication
#     publication_sentiments = {}

#     # Fetch unique publication names from the database
#     cursor.execute("SELECT DISTINCT name FROM news_items")
#     publications = cursor.fetchall()

#     # Iterate over each publication
#     for publication in publications:
#         publication_name = publication[0]

#         # Fetch the 10 newest articles for the current publication
#         cursor.execute("SELECT title FROM news_items WHERE name=? ORDER BY published_at DESC LIMIT 10", (publication_name,))
#         articles = cursor.fetchall()

#         # List to store sentiment predictions for the current publication
#         publication_sentiments[publication_name] = []

#         # Iterate over each article and predict sentiment
#         for article in articles:
#             title = article[0]
#             sentiment = predict_sentiment(title)
#             publication_sentiments[publication_name].append((title, sentiment))

#     return publication_sentiments

# # Predict sentiment for the 10 newest articles for each publication
# publication_sentiments = predict_sentiment_for_newest_articles()

# # Print sentiment predictions for each publication
# for publication, articles in publication_sentiments.items():
#     print("Publication:", publication)
#     for title, sentiment in articles:
#         print("Title:", title)
#         print("Sentiment:", sentiment)
#     print()


# %%
# import os
# from together import Together

# import requests

# endpoint = 'https://api.together.xyz/inference'
# TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

# res = requests.post(endpoint, json={
#     "model": 'meta-llama/Llama-3-70b-chat-hf',
#     "prompt": """\
#       Label the sentences as either "Positive", "Negative" or "Neutral":

#       Sentence: I can say that there isn't anything I would change.
#       Label: Positive

#       Sentence: I'm not sure about this.
#       Label: Neutral

#       Sentence: I think the background image could have been better.
#       Label: Negative

#       Sentence: I really like it.
#       Label:""",
#     "top_p": 1,
#     "top_k": 40,
#     "temperature": 0.8,
#     "max_tokens": 1,
#     "repetition_penalty": 1,
# }, headers={
#     "Authorization": f"Bearer {TOGETHER_API_KEY}",
#     "User-Agent": "<YOUR_APP_NAME>"
# })
# print(res.json()['output']['choices'][0]['text']) # ' positive'

# %%
# import os
# import requests
# import pandas as pd
# import sqlite3
# import pickle

# # Define the setup function
# def setup():
#     # Load your trained SVM model
#     with open("svm_classifier.pkl", "rb") as f:
#         svm_classifier = pickle.load(f)

#     # Load the TF-IDF vectorizer
#     with open("tfidf_vectorizer.pkl", "rb") as f:
#         vectorizer = pickle.load(f)

#     # Connect to SQLite database
#     conn = sqlite3.connect("news_database.db")
#     cursor = conn.cursor()

#     return svm_classifier, vectorizer, cursor

# # Call the setup function to get SVM classifier, TF-IDF vectorizer, and cursor
# svm_classifier, vectorizer, cursor = setup()

# # Function to predict sentiment for a given title using the SVM model
# def predict_sentiment_svm(title):
#     # Vectorize the title
#     title_vectorized = vectorizer.transform([title])

#     # Make prediction using SVM classifier
#     prediction = svm_classifier.predict(title_vectorized)

#     # Return the predicted sentiment
#     return prediction[0]

# # Function to predict sentiment for a given title using the Together model
# def predict_sentiment_together(title):
#     # Define the prompt
#     prompt = f"Label the sentence as either 'Positive', 'Negative', or 'Neutral':\n\nSentence: {title}\nLabel:"

#     # Make request to Together API
#     endpoint = 'https://api.together.xyz/inference'
#     TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

#     res = requests.post(endpoint, json={
#         "model": 'meta-llama/Llama-3-70b-chat-hf',
#         "prompt": prompt,
#         "top_p": 1,
#         "top_k": 40,
#         "temperature": 0.8,
#         "max_tokens": 1,
#         "repetition_penalty": 1,
#     }, headers={
#         "Authorization": f"Bearer {TOGETHER_API_KEY}",
#         "User-Agent": "<YOUR_APP_NAME>"
#     })

#     # Extract sentiment from response
#     sentiment = res.json()['output']['choices'][0]['text'].strip().lower()

#     return sentiment

# # Function to fetch the 10 newest articles for each publication and predict sentiment using both models
# def predict_sentiment_for_newest_articles():
#     # Dictionary to store sentiment predictions for each publication
#     publication_sentiments = {}

#     # Fetch unique publication names from the database
#     cursor.execute("SELECT DISTINCT name FROM news_items")
#     publications = cursor.fetchall()

#     # Iterate over each publication
#     for publication in publications:
#         publication_name = publication[0]

#         # Fetch the 10 newest articles for the current publication
#         cursor.execute("SELECT title FROM news_items WHERE name=? ORDER BY published_at DESC LIMIT 10", (publication_name,))
#         articles = cursor.fetchall()

#         # List to store sentiment predictions for the current publication using both models
#         publication_sentiments[publication_name] = []

#         # Iterate over each article and predict sentiment using both models
#         for article in articles:
#             title = article[0]
#             sentiment_svm = predict_sentiment_svm(title)
#             sentiment_together = predict_sentiment_together(title)
#             publication_sentiments[publication_name].append((title, sentiment_svm, sentiment_together))

#     return publication_sentiments

# # Predict sentiment for the 10 newest articles for each publication using both models
# publication_sentiments = predict_sentiment_for_newest_articles()

# # Print sentiment predictions for each publication
# for publication, articles in publication_sentiments.items():
#     print("Publication:", publication)
#     for title, sentiment_svm, sentiment_together in articles:
#         print("Title:", title)
#         print("Sentiment (SVM):", sentiment_svm)
#         print("Sentiment (Together):", sentiment_together)
#     print()


# %%
import os
import requests
import pandas as pd
import sqlite3
import pickle

# Define the setup function
def setup():
    # Load your trained SVM model
    with open("svm_classifier.pkl", "rb") as f:
        svm_classifier = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Connect to SQLite database
    conn = sqlite3.connect("news_database.db")
    cursor = conn.cursor()

    return svm_classifier, vectorizer, cursor

# Call the setup function to get SVM classifier, TF-IDF vectorizer, and cursor
svm_classifier, vectorizer, cursor = setup()

# Function to predict sentiment for a given title using the SVM model
def predict_sentiment_svm(title):
    # Vectorize the title
    title_vectorized = vectorizer.transform([title])

    # Make prediction using SVM classifier
    prediction = svm_classifier.predict(title_vectorized)

    # Return the predicted sentiment (capitalized)
    return prediction[0].capitalize()

# Function to predict sentiment for a given title using the Together model
def predict_sentiment_together(title):
    # Define the prompt
    prompt = f"Label the sentence as either 'Positive', 'Negative', or 'Neutral':\n\nSentence: {title}\nLabel:"

    # Make request to Together API
    endpoint = 'https://api.together.xyz/inference'
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    res = requests.post(endpoint, json={
        "model": 'meta-llama/Llama-3-70b-chat-hf',
        "prompt": prompt,
        "top_p": 1,
        "top_k": 40,
        "temperature": 0.8,
        "max_tokens": 1,
        "repetition_penalty": 1,
    }, headers={
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "User-Agent": "<YOUR_APP_NAME>"
    })

    # Extract sentiment from response
    sentiment = res.json()['output']['choices'][0]['text'].strip().capitalize()

    return sentiment

# Function to fetch the 10 newest articles for each publication and predict sentiment using both models
def predict_sentiment_for_newest_articles():
    # Dictionary to store sentiment predictions for each publication
    publication_sentiments = {}

    # Fetch unique publication names from the database
    cursor.execute("SELECT DISTINCT name FROM news_items")
    publications = cursor.fetchall()

    # Iterate over each publication
    for publication in publications:
        publication_name = publication[0]

        # Fetch the 10 newest articles for the current publication
        cursor.execute("SELECT title FROM news_items WHERE name=? ORDER BY published_at DESC LIMIT 10", (publication_name,))
        articles = cursor.fetchall()

        # List to store sentiment predictions for the current publication using both models
        publication_sentiments[publication_name] = []

        # Iterate over each article and predict sentiment using both models
        for article in articles:
            title = article[0]
            sentiment_svm = predict_sentiment_svm(title)
            sentiment_together = predict_sentiment_together(title)
            publication_sentiments[publication_name].append((title, sentiment_svm, sentiment_together))

    return publication_sentiments

# Predict sentiment for the 10 newest articles for each publication using both models
publication_sentiments = predict_sentiment_for_newest_articles()

# Print sentiment predictions for each publication
for publication, articles in publication_sentiments.items():
    print("Publication:", publication)
    for title, sentiment_svm, sentiment_together in articles:
        print("Title:", title)
        print("Sentiment (SVM):", sentiment_svm)
        print("Sentiment (Together):", sentiment_together)
    print()


# %%
import os
import requests
import pandas as pd
import sqlite3
import pickle

# Define the setup function
def setup():
    # Load your trained SVM model
    with open("svm_classifier.pkl", "rb") as f:
        svm_classifier = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Connect to SQLite database
    conn = sqlite3.connect("news_database.db")
    cursor = conn.cursor()

    return svm_classifier, vectorizer, cursor

# Call the setup function to get SVM classifier, TF-IDF vectorizer, and cursor
svm_classifier, vectorizer, cursor = setup()

# Function to predict sentiment for a given title using the SVM model
def predict_sentiment_svm(title):
    # Vectorize the title
    title_vectorized = vectorizer.transform([title])

    # Make prediction using SVM classifier
    prediction = svm_classifier.predict(title_vectorized)

    # Return the predicted sentiment (capitalized)
    return prediction[0].capitalize()

# Function to predict sentiment, political leaning, and bias for a given title using the Together model
def predict_sentiment_together(title):
    # Define the prompt
    prompt = f"""\
Label the sentence as either 'Positive', 'Negative', or 'Neutral', and indicate if the sentence is biased and what the political leaning is (Liberal, Conservative, Neutral):

Sentence: I can say that there isn't anything I would change.
Label: Positive, Not Biased, Neutral

Sentence: I'm not sure about this.
Label: Neutral, Not Biased, Neutral

Sentence: I think the background image could have been better.
Label: Negative, Not Biased, Neutral

Sentence: I really like it.
Label: Positive, Not Biased, Neutral

Sentence: The new policy will only benefit the wealthy.
Label: Negative, Biased, Liberal

Sentence: The government is taking necessary steps to help everyone.
Label: Positive, Biased, Conservative

Sentence: {title}
Label:"""

    # Make request to Together API
    endpoint = 'https://api.together.xyz/inference'
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    res = requests.post(endpoint, json={
        "model": 'meta-llama/Llama-3-70b-chat-hf',
        "prompt": prompt,
        "top_p": 1,
        "top_k": 40,
        "temperature": 0.8,
        "max_tokens": 50,  # Increased to handle longer responses
        "repetition_penalty": 1,
    }, headers={
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "User-Agent": "<YOUR_APP_NAME>"
    })

    # Extract sentiment, bias, and political leaning from response
    response_text = res.json()['output']['choices'][0]['text'].strip().split(", ")
    sentiment = response_text[0].capitalize()
    bias = response_text[1].capitalize()
    political_leaning = response_text[2].capitalize()

    return sentiment, bias, political_leaning

# Function to fetch the 10 newest articles for each publication and predict sentiment using both models
def predict_sentiment_for_newest_articles():
    # Dictionary to store sentiment predictions for each publication
    publication_sentiments = {}

    # Fetch unique publication names from the database
    cursor.execute("SELECT DISTINCT name FROM news_items")
    publications = cursor.fetchall()

    # Iterate over each publication
    for publication in publications:
        publication_name = publication[0]

        # Fetch the 10 newest articles for the current publication
        cursor.execute("SELECT title FROM news_items WHERE name=? ORDER BY published_at DESC LIMIT 10", (publication_name,))
        articles = cursor.fetchall()

        # List to store sentiment predictions for the current publication using both models
        publication_sentiments[publication_name] = []

        # Iterate over each article and predict sentiment using both models
        for article in articles:
            title = article[0]
            sentiment_svm = predict_sentiment_svm(title)
            sentiment_together, bias, political_leaning = predict_sentiment_together(title)
            publication_sentiments[publication_name].append((title, sentiment_svm, sentiment_together, bias, political_leaning))

    return publication_sentiments

# Predict sentiment for the 10 newest articles for each publication using both models
publication_sentiments = predict_sentiment_for_newest_articles()

# Print sentiment predictions for each publication
for publication, articles in publication_sentiments.items():
    print("Publication:", publication)
    for title, sentiment_svm, sentiment_together, bias, political_leaning in articles:
        print("Title:", title)
        print("Sentiment (SVM):", sentiment_svm)
        print("Sentiment (Together):", sentiment_together)
        print("Bias (Together):", bias)
        print("Political Leaning (Together):", political_leaning)
    print()


# %%
import os
import requests
import sqlite3
import pickle

# Define the setup function to load SVM model and TF-IDF vectorizer
def setup():
    # Load the SVM model
    with open("svm_classifier.pkl", "rb") as f:
        svm_classifier = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Connect to SQLite database
    conn = sqlite3.connect("news_database.db")
    cursor = conn.cursor()

    return svm_classifier, vectorizer, cursor

# Call the setup function to get SVM classifier, TF-IDF vectorizer, and cursor
svm_classifier, vectorizer, cursor = setup()

# Function to predict sentiment for a given title using the SVM model
def predict_sentiment_svm(title):
    # Vectorize the title
    title_vectorized = vectorizer.transform([title])

    # Make prediction using SVM classifier
    prediction = svm_classifier.predict(title_vectorized)

    # Return the predicted sentiment (capitalized)
    return prediction[0].capitalize()

# Function to predict sentiment, political leaning, and bias for a given title using the Together model
def predict_sentiment_together(title):
    # Define the prompt for the Together API
    prompt = f"""\
Label the news headline as either 'Positive', 'Negative', or 'Neutral', and indicate if the sentence is biased and what the political leaning is (Liberal, Conservative, Neutral):

Headline: {title}
Label:"""

    # Make request to Together API
    endpoint = 'https://api.together.xyz/inference'
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    res = requests.post(endpoint, json={
        "model": 'meta-llama/Llama-3-70b-chat-hf',
        "prompt": prompt,
        "top_p": 1,
        "top_k": 40,
        "temperature": 0.8,
        "max_tokens": 50,  # Increased to handle longer responses
        "repetition_penalty": 1,
    }, headers={
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "User-Agent": "<YOUR_APP_NAME>"
    })

    # Extract sentiment, bias, and political leaning from response
    response_choices = res.json()['output']['choices']
    for choice in response_choices:
        text = choice['text'].strip()
        if text.startswith("Positive"):
            sentiment = "Positive"
        elif text.startswith("Negative"):
            sentiment = "Negative"
        elif text.startswith("Neutral"):
            sentiment = "Neutral"
        else:
            sentiment = "Unknown"

        if "Biased" in text:
            bias = "Biased"
        else:
            bias = "Not Biased"

        if "Liberal" in text:
            political_leaning = "Liberal"
        elif "Conservative" in text:
            political_leaning = "Conservative"
        elif "Neutral" in text:
            political_leaning = "Neutral"
        else:
            political_leaning = "Unknown"

        if sentiment != "Unknown" and bias != "Unknown" and political_leaning != "Unknown":
            break  # Exit loop if all values are found

    return sentiment, bias, political_leaning

# Function to fetch the 10 newest articles and predict sentiment using both models
def predict_sentiment_for_newest_articles(max_articles=3):
    # List to store sentiment predictions
    all_articles_sentiments = []

    # Fetch the newest articles from the database
    cursor.execute("SELECT title FROM news_items ORDER BY published_at DESC LIMIT ?", (max_articles,))
    articles = cursor.fetchall()

    # Iterate over each article and predict sentiment using both models
    for article in articles:
        title = article[0]
        sentiment_svm = predict_sentiment_svm(title)
        sentiment_together, bias, political_leaning = predict_sentiment_together(title)
        all_articles_sentiments.append((title, sentiment_svm, sentiment_together, bias, political_leaning))

    return all_articles_sentiments

# Predict sentiment for the newest articles using both models
articles_sentiments = predict_sentiment_for_newest_articles()

# Print sentiment predictions
for title, sentiment_svm, sentiment_together, bias, political_leaning in articles_sentiments:
    print("Title:", title)
    print("Sentiment (SVM):", sentiment_svm)
    print("Sentiment (Together):", sentiment_together)
    print("Bias (Together):", bias)
    print("Political Leaning (Together):", political_leaning)
    print()


# %%



