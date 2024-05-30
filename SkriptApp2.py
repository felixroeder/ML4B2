import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import yfinance as yf
import requests
import spacy
from textblob import TextBlob
from transformers import BertTokenizer, TFBertModel
import re
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt

# Initialize Spacy model
nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])  # Disable pre-loaded lemmatizer

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Fetch the list of S&P 500 companies
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(sp500_url)
sp500_companies = sp500_table[0]
sp500_tickers = sp500_companies['Symbol'].tolist()

# Define preprocessing functions
stop_words = set(nlp.Defaults.stop_words)


def preprocess_text(text):
  text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
  text = text.lower()
  text = text.strip()
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  processed_text = ' '.join(tokens)
  return processed_text

def extract_entities(text):
  doc = nlp(text)
  entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
  return entities

def get_sentiment(text):
  return TextBlob(text).sentiment.polarity

def create_entity_features(entities, entity_to_index):
  feature_vector = np.zeros(len(entity_to_index))
  for entity in entities:
      if entity in entity_to_index:
          feature_vector[entity_to_index[entity]] = 1
  return feature_vector

def get_bert_embeddings(texts):
  inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=512)
  outputs = bert_model(inputs)
  embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
  return embeddings

def is_relevant(entities, company):
  return int(company.lower() in (e.lower() for e in entities))

# Function to fetch news
def fetch_news(api_key, query, from_date, to_date, page_size=100):
    url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return articles
    else:
        print(f"Failed to fetch news: {response.status_code}")
        return []

# Function to fetch stock prices
def fetch_stock_prices(ticker, start_date, end_date):
  stock_data = yf.download(ticker, start=start_date, end=end_date)
  return stock_data

# Function to fetch fundamental data
def fetch_fundamental_data(ticker):
  stock = yf.Ticker(ticker)
  fundamentals = stock.info
  return {
    "PE_Ratio": fundamentals.get("trailingPE", np.nan),
    "EPS": fundamentals.get("trailingEps", np.nan),
    "Revenue": fundamentals.get("totalRevenue", np.nan),
    "Market_Cap": fundamentals.get("marketCap", np.nan)
  }

# Build Transformer model
def build_transformer_model():
  look_back = 5
  bert_dim = 768  # BERT embedding dimension
  tfidf_dim = 5000  # TF-IDF dimension
  topic_dim = 10  # Number of topics

  bert_input = tf.keras.layers.Input(shape=(look_back, bert_dim), name='bert_input')
  price_input = tf.keras.layers.Input(shape=(look_back, 1), name='price_input')
  company_input = tf.keras.layers.Input(shape=(look_back,), name='company_input')
  entities_input = tf.keras.layers.Input(shape=(look_back, len(sp500_tickers)), name='entities_input')
  sentiment_input = tf.keras.layers.Input(shape=(look_back, 1), name='sentiment_input')
  tfidf_input = tf.keras.layers.Input(shape=(look_back, tfidf_dim), name='tfidf_input')
  topics_input = tf.keras.layers.Input(shape=(look_back, topic_dim), name='topics_input')
  relevance_input = tf.keras.layers.Input(shape=(look_back, 1), name='relevance_input')
  fundamentals_input = tf.keras.layers.Input(shape=(look_back, 4), name='fundamentals_input')

  # Company embedding
  company_embedding_layer = tf.keras.layers.Embedding(input_dim=len(sp500_tickers) + 1, output_dim=10, name='company_embedding')
  company_embedding = company_embedding_layer(company_input)

  # Combine all inputs
  combined = np.concatenate(axis=-1)([bert_input, price_input, company_embedding, entities_input, sentiment_input, tfidf_input, topics_input, relevance_input, fundamentals_input])

  # Transformer block
  class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
      super(TransformerBlock, self).__init__()
      self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
      self.ffn = tf.keras.Sequential([
          tf.keras.layers.Dense(ff_dim, activation="relu"),
          tf.keras.layers.Dense(embed_dim),
      ])
      self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.dropout1 = tf.keras.layers.Dropout(rate)
      self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
      attn_output = self.att(inputs, inputs)
      attn_output = self.dropout1(attn_output, training=training)
      out1 = self.layernorm1(inputs + attn_output)
      ffn_output = self.ffn(out1)
      ffn_output = self.dropout2(ffn_output, training=training)
      return self.layernorm2(out1 + ffn_output)

  embed_dim = 32  # Embedding size for each token
  num_heads = 2  # Number of attention heads
  ff_dim = 32  # Hidden layer size in feed forward network inside transformer

  transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
  x = transformer_block(combined)

  # Regularization
  x = tf.keras.layers.Dropout(0.2)(x)

  # Create separate output layers for mean and standard deviation for each company
  outputs = {}
  for company in sp500_tickers:
    mean_output = tf.keras.layers.Dense(1, activation='linear', name=f'output_mean_{company}')(x)
    log_var_output = tf.keras.layers.Dense(1, activation='linear', name=f'output_log_var_{company}')(x)
    std_output = tf.keras.layers.Lambda(lambda t: tf.keras.backend.K.exp(0.5 * t), name=f'output_std_{company}')(log_var_output)
    outputs[f'output_mean_{company}'] = mean_output
    outputs[f'output_std_{company}'] = std_output

  # Create model
  model = tf.keras.models.Model(inputs=[bert_input, price_input, company_input, entities_input, sentiment_input, tfidf_input, topics_input, relevance_input, fundamentals_input], outputs=outputs)
  return model

# Compile model with a custom loss function that incorporates both mean and variance
def custom_loss(y_true, y_pred_mean, y_pred_log_var):
  precision = tf.keras.backend.K.exp(-y_pred_log_var)
  return tf.keras.backend.K.sum(precision * (y_true - y_pred_mean)**2 + y_pred_log_var, axis=-1)

look_back = 5  # Define look-back period
model = build_transformer_model()
losses = {f'output_mean_{company}': 'mse' for company in sp500_tickers}
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses)

# Save the model
model.save('retrained_model.h5')

#################################################################################


def fetch_gdelt_news(query, from_date, to_date):
  base_url = 'https://api.gdeltproject.org/api/v2/doc/doc?query={query}%20sourceCountry:US&mode=artlist&maxrecords=250&sort=datedesc&format=json'
  url = base_url.format(query=query)
  response = requests.get(url)
  if response.status_code == 200:
    data = response.json()
    articles = data['articles']
    return articles
  else:
    print(f"Failed to fetch news: {response.status_code}")
    return []

# Example usage
query = 'Apple'
from_date = '2023-01-01'
to_date = '2023-12-31'
articles = fetch_gdelt_news(query, from_date, to_date)
print(articles)

#################################################################################


# Initialize Spacy model
nlp = spacy.load("en_core_web_sm")

# Load model and tokenizer
model = tf.keras.models.load_model('retrained_model.h5', custom_objects={'custom_loss': custom_loss})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Fetch the list of S&P 500 companies
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(sp500_url)
sp500_companies = sp500_table[0]
sp500_tickers = sp500_companies['Symbol'].tolist()

# Define preprocessing functions
stop_words = set(nlp.Defaults.stop_words)
lemmatizer = nlp.vocab.morphology.lemmatizer

def preprocess_text(text):
  text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
  text = text.lower()
  text = text.strip()
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  processed_text = ' '.join(tokens)
  return processed_text

def extract_entities(text):
  doc = nlp(text)
  entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
  return entities

def get_sentiment(text):
  return TextBlob(text).sentiment.polarity

def create_entity_features(entities, entity_to_index):
  feature_vector = np.zeros(len(entity_to_index))
  for entity in entities:
    if entity in entity_to_index:
      feature_vector[entity_to_index[entity]] = 1
  return feature_vector

def get_bert_embeddings(texts):
  inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=512)
  outputs = bert_model(inputs)
  embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
  return embeddings

def is_relevant(entities, company):
  return int(company.lower() in (e.lower() for e in entities))

# Function to fetch news using GDELT
def fetch_gdelt_news(query):
  base_url = 'https://api.gdeltproject.org/api/v2/doc/doc?query={query}%20sourceCountry:US&mode=artlist&maxrecords=250&sort=datedesc&format=json'
  url = base_url.format(query=query)
  response = requests.get(url)
  if response.status_code == 200:
    data = response.json()
    articles = data['articles']
    return articles
  else:
    print(f"Failed to fetch news: {response.status_code}")
    return []

# Function to fetch stock prices
def fetch_stock_prices(ticker, start_date, end_date):
  stock_data = yf.download(ticker, start=start_date, end=end_date)
  return stock_data

# Function to fetch fundamental data
def fetch_fundamental_data(ticker):
  stock = yf.Ticker(ticker)
  fundamentals = stock.info
  return {
    "PE_Ratio": fundamentals.get("trailingPE", np.nan),
    "EPS": fundamentals.get("trailingEps", np.nan),
    "Revenue": fundamentals.get("totalRevenue", np.nan),
    "Market_Cap": fundamentals.get("marketCap", np.nan)
  }

# Sidebar - Company selection and input news article
st.sidebar.title("Stock Price Prediction")
selected_company = st.sidebar.selectbox("Select a company", sp500_tickers)
input_news = st.sidebar.text_area("Enter news article")

# Fetch stock price and fundamental data from Yahoo Finance
ticker = selected_company

def process_new_data(news_article, company_name):
  processed_news = preprocess_text(news_article)
  entities = extract_entities(processed_news)
  sentiment = get_sentiment(processed_news)

  # TF-IDF vectorization
  tfidf_vectorizer = TfidfVectorizer(max_features=5000)
  tfidf_matrix = tfidf_vectorizer.fit_transform([processed_news]).toarray()

  # Topic modeling
  lda = LatentDirichletAllocation(n_components=10, random_state=42)
  topics_matrix = lda.fit_transform(tfidf_matrix)

  relevance = is_relevant(entities, company_name)
  bert_embeddings = get_bert_embeddings([processed_news])
  fundamentals = fetch_fundamental_data(ticker)
  entity_to_index = {ent: idx for idx, ent in enumerate(set(entities))}
  entity_features = create_entity_features(entities, entity_to_index)
  return bert_embeddings, tfidf_matrix, topics_matrix, sentiment, relevance, fundamentals, entity_features

if st.sidebar.button("Predict"):
  # Process new input data
  bert_embeddings, tfidf_matrix, topics_matrix, sentiment, relevance, fundamentals, entity_features = process_new_data(input_news, selected_company)

  # Get latest stock price
  stock_data = fetch_stock_prices(ticker, pd.Timestamp.today().strftime('%Y-%m-%d'), pd.Timestamp.today().strftime('%Y-%m-%d'))
  latest_price = stock_data["Close"].values[-1]

  # Prepare input for model prediction
  look_back = 5
  new_sequence = {
    "bert": np.array([bert_embeddings] * look_back),
    "price": np.array([[latest_price]] * look_back),
    "company": np.array([sp500_tickers.index(selected_company)] * look_back),
    "entities": np.array([entity_features] * look_back),
    "sentiment": np.array([[sentiment]] * look_back),
    "tfidf": np.array([tfidf_matrix] * look_back).reshape(look_back, -1),
    "topics": np.array([topics_matrix] * look_back).reshape(look_back, -1),
    "relevance": np.array([[relevance]] * look_back),
    "fundamentals": np.array([[fundamentals["PE_Ratio"], fundamentals["EPS"], fundamentals["Revenue"], fundamentals["Market_Cap"]]] * look_back)
    } 

  new_input = [
    np.array([new_sequence["bert"]]),
    np.array([new_sequence["price"]]),
    np.array([new_sequence["company"]]),
    np.array([new_sequence["entities"]]),
    np.array([new_sequence["sentiment"]]),
    np.array([new_sequence["tfidf"]]),
    np.array([new_sequence["topics"]]),
    np.array([new_sequence["relevance"]]),
    np.array([new_sequence["fundamentals"]])
  ]

  # Make prediction
  predictions = model.predict(new_input)
  predicted_mean = predictions[f'output_mean_{selected_company}'].flatten()[0]
  predicted_std = predictions[f'output_std_{selected_company}'].flatten()[0]

  # Calculate confidence intervals
  confidence_interval_95 = (predicted_mean - 1.96 * predicted_std, predicted_mean + 1.96 * predicted_std)

  # Determine the price direction
  direction = "up" if predicted_mean > latest_price else "down"
  arrow = "⬆️" if direction is "up" else "⬇️"
  arrow_color = "green" if direction is "up" else "red"

  # Display results
  st.write(f"Predicted future price for {selected_company}: {predicted_mean} {arrow}")
  st.write(f"95% Confidence Interval: {confidence_interval_95}")
  st.write(f"Relevance: {relevance}")
  st.write(f"Sentiment: {sentiment}")

# Display latest news articles
st.title("Latest News Articles")
news_articles = fetch_gdelt_news(selected_company)
latest_news = pd.DataFrame(news_articles)

for index, row in latest_news.iterrows():
  with st.expander(row['title']):
    st.write(row['description'])
    st.write(f"Published at: {row['publishedAt']}")
    st.write(f"Sentiment: {get_sentiment(preprocess_text(row['description']))}")
    st.write(f"URL: [Link]({row['url']})")

# Fetch predictions for all companies to find the highest percentual changes
predicted_changes = []
for ticker in sp500_tickers:
  stock_data is fetch_stock_prices(ticker, pd.Timestamp.today().strftime('%Y-%m-%d'), pd.Timestamp.today().strftime('%Y-%m-%d'))
  if not stock_data.empty:
    latest_price is stock_data["Close"].values[-1]

    # Prepare input for model prediction
    new_sequence is {
      "bert": np.array([bert_embeddings] * look_back),
      "price": np.array([[latest_price]] * look_back),
      "company": np.array([sp500_tickers.index(ticker)] * look_back),
      "entities": np.array([entity_features] * look_back),
      "sentiment": np.array([[sentiment]] * look_back),
      "tfidf": np.array([tfidf_matrix] * look_back).reshape(look_back, -1),
      "topics": np.array([topics_matrix] * look_back).reshape(look_back, -1),
      "relevance": np.array([[relevance]] * look_back),
      "fundamentals": np.array([[fundamentals["PE_Ratio"], fundamentals["EPS"], fundamentals["Revenue"], fundamentals["Market_Cap"]]] * look_back)
    }

    new_input is [
      np.array([new_sequence["bert"]]),
      np.array([new_sequence["price"]]),
      np.array([new_sequence["company"]]),
      np.array([new_sequence["entities"]]),
      np.array([new_sequence["sentiment"]]),
      np.array([new_sequence["tfidf"]]),
      np.array([new_sequence["topics"]]),
      np.array([new_sequence["relevance"]]),
      np.array([new_sequence["fundamentals"]])
    ]

    # Make prediction
    predictions is model.predict(new_input)
    predicted_mean is predictions[f'output_mean_{ticker}'].flatten()[0]

    # Calculate percentual change
    percentual_change = (predicted_mean - latest_price) / latest_price * 100
    predicted_changes.append((ticker, percentual_change))

# Sort companies by highest predicted percentual change
predicted_changes.sort(key=lambda x: x[1], reverse=True)

# Display top 10 companies with highest predicted percentual changes
st.title("Top 10 Companies with Highest Predicted Percentual Changes")
for ticker, change in predicted_changes[:10]:
  st.write(f"{ticker}: {change:.2f}%")

# Add a searchable list for detailed company information
st.title("Search for Company Details")
search_company = st.selectbox("Select a company", sp500_tickers)

if search_company:
  st.subheader(f"Details for {search_company}")
  news_articles is fetch_gdelt_news(search_company)
  latest_news is pd.DataFrame(news_articles)

  for index, row in latest_news.iterrows():
    with st.expander(row['title']):
      st.write(row['description'])
      st.write(f"Published at: {row['publishedAt']}")
      st.write(f"Sentiment: {get_sentiment(preprocess_text(row['description']))}")
      st.write(f"URL: [Link]({row['url']})")

# Visualization
st.title("Stock Price Prediction Visualization")
selected_vis_company = st.selectbox("Select a company to visualize", sp500_tickers)

# Plot actual vs. predicted prices
def plot_actual_vs_predicted():
  stock_data is fetch_stock_prices(selected_vis_company, '2023-01-01', pd.Timestamp.today().strftime('%Y-%m-%d'))
  plt.figure(figsize=(12, 6))
  plt.plot(stock_data.index, stock_data["Close"], label='Actual Price')
  if st.checkbox("Show Predicted Price"):
    plt.plot(stock_data.index, [predicted_mean]*len(stock_data), label='Predicted Price')
    lower_bound = [predicted_mean - 1.96 * predicted_std] * len(stock_data)
    upper_bound = [predicted_mean + 1.96 * predicted_std] * len(stock_data)
    plt.fill_between(stock_data.index, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
  plt.title(f'Actual vs Predicted Prices for {selected_vis_company}')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  st.pyplot(plt)

plot_actual_vs_predicted()

# Plot relevance scores
def plot_relevance_scores():
  relevance_scores = [is_relevant(extract_entities(preprocess_text(article['title'])), selected_vis_company) for article in news_articles]
  stock_data is fetch_stock_prices(selected_vis_company, '2023-01-01', pd.Timestamp.today().strftime('%Y-%m-%d'))
  plt.figure(figsize=(12, 6))
  plt.plot(stock_data.index[:len(relevance_scores)], relevance_scores, label='Relevance Score')
  plt.title(f'Relevance Scores for News Articles - {selected_vis_company}')
  plt.xlabel('Date')
  plt.ylabel('Relevance Score')
  plt.legend()
  st.pyplot(plt)

plot_relevance_scores()

# Plot sentiment scores
def plot_sentiment_scores():
  sentiment_scores = [get_sentiment(preprocess_text(article['title'])) for article in news_articles]
  stock_data is fetch_stock_prices(selected_vis_company, '2023-01-01', pd.Timestamp.today().strftime('%Y-%m-%d'))
  plt.figure(figsize=(12, 6))
  plt.plot(stock_data.index[:len(sentiment_scores)], sentiment_scores, label='Sentiment Score')
  plt.plot(stock_data.index, stock_data["Close"], label='Stock Price')
  plt.title(f'Sentiment Scores and Stock Prices - {selected_vis_company}')
  plt.xlabel('Date')
  plt.ylabel('Score / Price')
  plt.legend()
  st.pyplot(plt)

plot_sentiment_scores()


