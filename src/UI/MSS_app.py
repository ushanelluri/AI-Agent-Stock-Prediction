import os
import sys
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.sma import SMAIndicator

st.title("Prototype Trading System")

symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

data_fetcher = DataFetcher()
data = data_fetcher.get_stock_data(symbol)

st.write(f"Original Stock Data for {symbol}:")
st.dataframe(data.tail())

if st.button("Calculate SMA"):
    period = st.number_input("Enter SMA period:", min_value=1, max_value=100, value=14)
    sma_indicator = SMAIndicator(period=period)
    data_with_sma = sma_indicator.calculate(data)
    st.write(f"Stock Data with SMA{period} for {symbol}:")
    st.dataframe(data_with_sma.tail())

if st.button("Calculate RSI"):
    period = st.number_input("Enter RSI period:", min_value=1, max_value=100, value=14)
    data[f"RSI{period}"] = ta.rsi(data['Close'], length=period)
    st.write(f"Stock Data with RSI{period} for {symbol}:")
    st.dataframe(data.tail())

class DataCollectionAgent:
    def fetch_news_articles(self, symbol):
        url = f"https://finnhub.io/api/v1/news?category=general&token=cuik1g1r01qtqfmisj9gcuik1g1r01qtqfmisja0"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()
            return articles[:5]
        return []

    def collect_social_media_data(self, symbol):
        sample_posts = [
            f"{symbol} is on the rise! 🚀 #bullish",
            f"Worried about {symbol}'s volatility today...",
            f"Market analysts predict a strong quarter for {symbol}.",
        ]
        return sample_posts

    def get_financial_reports(self, symbol):
        stock = yf.Ticker(symbol)
        return stock.financials

    def preprocess_data(self, data):
        return data.dropna()

    def exploratory_data_analysis(self, data):
        plt.figure(figsize=(8, 4))
        sns.histplot(data['Close'], bins=30, kde=True)
        st.pyplot(plt)

class SentimentAnalysisAgent:
    def analyze_text_sentiment(self, text):
        sentiment_score = TextBlob(text).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        return sentiment_score, sentiment

    def extract_sentiment_features(self, texts):
        return [TextBlob(text).sentiment.polarity for text in texts]

    def train_ml_model(self, data, labels):
        vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X_counts = vectorizer.fit_transform(data)
        X_tfidf = tfidf_transformer.fit_transform(X_counts)
        clf = MultinomialNB().fit(X_tfidf, labels)
        return clf, vectorizer, tfidf_transformer

    def train_dl_model(self, data, labels, max_words=5000, max_len=100):
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        X = pad_sequences(sequences, maxlen=max_len)

        model = Sequential([
            Embedding(max_words, 128, input_length=max_len),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, labels, epochs=5, batch_size=32, verbose=1)
        return model, tokenizer

    def implement_model_stacking(self, data, labels):
        vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X_counts = vectorizer.fit_transform(data)
        X_tfidf = tfidf_transformer.fit_transform(X_counts)

        base_models = [
            ('nb', MultinomialNB()),
            ('svm', SVC(probability=True))
        ]
        stacked_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
        stacked_model.fit(X_tfidf, labels)
        return stacked_model, vectorizer, tfidf_transformer

    def analyze_news_sentiment(self, articles):
        results = []
        for article in articles:
            score, sentiment = self.analyze_text_sentiment(article["summary"])
            results.append({"headline": article["headline"], "sentiment": sentiment, "score": score})
        return results

    def analyze_social_media_sentiment(self, posts):
        results = []
        for post in posts:
            score, sentiment = self.analyze_text_sentiment(post)
            results.append({"post": post, "sentiment": sentiment, "score": score})
        return results

data_agent = DataCollectionAgent()
sentiment_agent = SentimentAnalysisAgent()

if st.button("Fetch News Articles"):
    articles = data_agent.fetch_news_articles(symbol)
    for article in articles:
        st.write(f"**{article['headline']}** - {article['source']}")
        st.write(article['summary'])
        st.write("---")

if st.button("Analyze News Sentiment"):
    articles = data_agent.fetch_news_articles(symbol)
    sentiment_results = sentiment_agent.analyze_news_sentiment(articles)
    for result in sentiment_results:
        st.write(f"**{result['headline']}** - Sentiment: {result['sentiment']} (Score: {result['score']:.2f})")
        st.write("---")

if st.button("Analyze Social Media Sentiment"):
    posts = data_agent.collect_social_media_data(symbol)
    sentiment_results = sentiment_agent.analyze_social_media_sentiment(posts)
    sentiment_df = pd.DataFrame(sentiment_results)
    st.write("Social Media Sentiment Analysis:")
    st.dataframe(sentiment_df)
    st.bar_chart(sentiment_df.set_index("post")["score"])

if st.button("Train ML Model"):
    data_samples = ["The market is up!", "Stocks are falling", "Stable outlook"]
    labels = [1, 0, 1]
    ml_model, vectorizer, tfidf = sentiment_agent.train_ml_model(data_samples, labels)
    st.write("Machine Learning Model Trained Successfully!")

if st.button("Train DL Model"):
    data_samples = ["The stock is bullish", "Bearish trend continues", "Investors are optimistic"]
    labels = [1, 0, 1]
    dl_model, tokenizer = sentiment_agent.train_dl_model(data_samples, labels)
    st.write("Deep Learning Model Trained Successfully!")

if st.button("Implement Model Stacking"):
    data_samples = ["Bullish trend expected", "Stock market crashes", "Positive earnings report"]
    labels = [1, 0, 1]
    stacked_model, vectorizer, tfidf = sentiment_agent.implement_model_stacking(data_samples, labels)
    st.write("Model Stacking Implemented Successfully!")
