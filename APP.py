import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from transformers import RobertaTokenizer, TFRobertaModel
import plotly.graph_objs as go
from textblob import TextBlob
import re

# Define the company tickers and names
companies_to_focus = {
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'AAPL': 'Apple'
}

# Initialize tokenizer and BERT model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = TFRobertaModel.from_pretrained('roberta-base')

# Define lookback window
look_back = 5

# Register the custom layer for deserialization
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
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

# Load the trained model with the custom layer
custom_objects = {'TransformerBlock': TransformerBlock}
model = tf.keras.models.load_model('model.keras', custom_objects=custom_objects)

# Function to preprocess text for BERT embeddings
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower().strip()
    tokens = text.split()
    return ' '.join(tokens)

# Function to get BERT embeddings
def get_bert_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token's embedding

# Define dimensions
bert_dim = bert_model.config.hidden_size  # typically 768 for BERT models
combined_dim = 6  # Update this to match the model's expected input dimension

# Function to predict future prices
def predict_prices(news_headlines, look_back_window, bert_dim, combined_dim):
    processed_articles = [preprocess_text(article) for article in news_headlines]
    bert_embeddings = [get_bert_embeddings([article], tokenizer, bert_model)[0] for article in processed_articles]

    # Ensure the embeddings have the correct shape and dimension
    bert_embeddings = bert_embeddings[-look_back_window:]
    if len(bert_embeddings) < look_back_window:
        # Pad the embeddings if there are not enough look-back days
        padding = [np.zeros((bert_dim,)) for _ in range(look_back_window - len(bert_embeddings))]
        bert_embeddings = padding + bert_embeddings

    # Convert to numpy array
    bert_embeddings = np.array(bert_embeddings)

    # Create combined_features with shape (look_back_window, combined_dim)
    combined_features = np.zeros((look_back_window, combined_dim))

    # Fill combined_features with truncated or extended bert_embeddings
    for i in range(look_back_window):
        combined_features[i, :min(combined_dim, bert_dim)] = bert_embeddings[i, :combined_dim]

    # Ensure the embeddings have the correct shape
    combined_features = combined_features.reshape(1, look_back_window, combined_dim)

    # Predict using the loaded model
    predictions = model.predict(combined_features)
    return predictions


# Function to perform sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to fetch fundamental data for a company
def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamentals = stock.info
    return {
        "PE_Ratio": fundamentals.get("trailingPE", np.nan),
        "EPS": fundamentals.get("trailingEps", np.nan),
        "Revenue": fundamentals.get("totalRevenue", np.nan),
        "Market_Cap": fundamentals.get("marketCap", np.nan)
    }

# Load the dataset
news_data = pd.read_csv('Datensatz.csv')
news_data['Date'] = pd.to_datetime(news_data['Date'])
news_data['Processed_Article'] = news_data['News Article'].apply(preprocess_text)
news_data['Sentiment'] = news_data['Processed_Article'].apply(get_sentiment)

# Streamlit App Layout
st.title("Stock Price Prediction App")

# Fetch data
today = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=look_back * 2)).strftime('%Y-%m-%d')
end_date = today

# Get today's news headlines
todays_news = news_data[news_data['Date'] == today]

# Define dimensions
bert_dim = bert_model.config.hidden_size  # typically 768 for BERT models
#combined_dim = 1543  # Update this to the correct combined dimension

# Get stock data and predictions
stock_data_dict = {}
fundamental_data_dict = {}
for ticker in companies_to_focus:
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate moving averages
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

    stock_data_dict[ticker] = stock_data
    fundamental_data_dict[ticker] = fetch_fundamental_data(ticker)

# Call predict_prices once
news_headlines = todays_news['Processed_Article'].tolist()
predictions = predict_prices(news_headlines, look_back, bert_dim, combined_dim)

# Initialize predictions_dict
predictions_dict = {}

# Ensure predictions is reshaped or processed correctly based on model output
predictions = np.squeeze(predictions)  # Flatten predictions if necessary

# Populate predictions_dict with the single prediction for each ticker
for ticker in companies_to_focus:
    predictions_dict[ticker] = predictions  # Store the single prediction value

# Display predicted prices
st.subheader("Predicted Prices for Tomorrow")
for ticker, company in companies_to_focus.items():
    today_price = stock_data_dict[ticker]['Close'].values[-1]
    
    # Fetch the prediction for the specific ticker
    predicted_price = predictions_dict[ticker]
    
    # Handle single value or array of predictions
    if isinstance(predicted_price, np.ndarray):
        if predicted_price.size == 1:
            predicted_price = predicted_price.item()  # Convert single-element numpy array to scalar
        else:
            predicted_price = predicted_price[0]  # Take the first element if it's an array
        
    # Determine arrow and color based on comparison
    if isinstance(predicted_price, (int, float)):
        if predicted_price > today_price:
            arrow = "⬆️"
            color = "green"
        else:
            arrow = "⬇️"
            color = "red"
    else:
        arrow = "↔️"
        color = "gray"
    
    st.markdown(f"**{company} ({ticker}):** {predicted_price:.2f} {arrow}", unsafe_allow_html=True)

# Display news headlines with sentiment in a table
st.subheader("Latest News")
news_table = todays_news[['News Article', 'Sentiment']].copy()
news_table['Sentiment_Color'] = news_table['Sentiment'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray')
news_table['Sentiment_Display'] = news_table.apply(lambda row: f"<span style='color:{row['Sentiment_Color']}'>{row['Sentiment']:.2f}</span>", axis=1)
st.write(news_table[['News Article', 'Sentiment_Display']].to_html(escape=False, index=False), unsafe_allow_html=True)

# Manual prediction input
st.subheader("Manual Prediction")
manual_input = st.text_input("Enter news headline for manual prediction")
manual_look_back = st.slider("Look Back Window", min_value=1, max_value=30, value=look_back)

if manual_input:
    manual_prediction = predict_prices([manual_input], manual_look_back, bert_dim, combined_dim)

    # Debugging: Inspect the structure of manual_prediction
    st.write("Debugging: Structure of manual_prediction")
    st.write(manual_prediction)

    # Access the prediction correctly
    for ticker in companies_to_focus:
        st.write(f"Predicted price for {ticker}: {manual_prediction[ticker][0][0]}")

# Display stock price charts with predicted prices
st.subheader("Stock Prices")
for ticker, company in companies_to_focus.items():
    stock_data = stock_data_dict[ticker]
    fig = go.Figure()

    # Add actual stock price trace
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Close'))

    # Add predicted price trace
    predicted_price = predictions_dict[ticker]
    predicted_date = stock_data.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(x=[predicted_date], y=[predicted_price], mode='markers', name='Predicted Close', marker=dict(color='red', size=10)))

    # Add moving average traces
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='MA50'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='MA200'))

    # Customize the layout
    fig.update_layout(
        title=f'{company} ({ticker}) Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True
    )

    # Display the chart
    st.plotly_chart(fig)

    # Display fundamental data
    st.subheader(f"{company} ({ticker}) Fundamentals")
    fundamentals = fundamental_data_dict[ticker]
    st.markdown(f"""
    - **PE Ratio**: {fundamentals['PE_Ratio']}
    - **EPS**: {fundamentals['EPS']}
    - **Revenue**: {fundamentals['Revenue']}
    - **Market Cap**: {fundamentals['Market_Cap']}
    """)
