import streamlit as st
import yfinance as yf
import newsapi
from transformers import BertTokenizer
import tensorflow as tf

# Function to preprocess text
def preprocess_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    return inputs
print("Hallo1")
model = tf.keras.models.load_model('trained_model.h5', safe_mode=False, compile=False)
print("Hallo2")

# Initialize NewsAPI client
api = newsapi.NewsApiClient(api_key='ae91264b58784ff19f181a6691c1efc6')

# Define companies and stock tickers
companies = {
    'Apple': 'AAPL',
    'Amazon': 'AMZN',
    'Google': 'GOOGL'
}

# Function to fetch latest stock prices
def fetch_latest_stock_prices():
    stock_data = {}
    for company, ticker in companies.items():
        data = yf.download(ticker, period='1d')
        stock_data[company] = data['Close'][-1]
    return stock_data

# Function to fetch latest news articles
def fetch_latest_news():
    news_data = {}
    for company in companies:
        articles = api.get_everything(q=company, language='en', sort_by='publishedAt', page_size=5)
        news_data[company] = articles['articles']
    return news_data

# Function to predict stock prices
def predict_stock_prices(preprocessed_news):
    predictions = {}
    for company in companies:
        # Assuming preprocessed_news is a dictionary with preprocessed news articles for each company
        inputs = preprocessed_news[company]
        outputs = model.predict(inputs)
        predictions[company] = outputs['output_' + company][0][0]
    return predictions

# Streamlit app layout
st.title("Stock Price Prediction")

# Fetch latest stock prices and news
latest_prices = fetch_latest_stock_prices()
latest_news = fetch_latest_news()

# Display predicted prices
preprocessed_news = {company: preprocess_text(" ".join([article['title'] for article in articles])) for company, articles in latest_news.items()}
predicted_prices = predict_stock_prices(preprocessed_news)

st.write("## Predicted Prices for Tomorrow")
for company, price in predicted_prices.items():
    arrow = "⬆️" if price > latest_prices[company] else "⬇️"
    color = "green" if price > latest_prices[company] else "red"
    st.markdown(f"<span style='color:{color};'>{arrow} {company}: {price:.2f}</span>", unsafe_allow_html=True)

# Display latest news articles
st.write("## Latest News")
for company, articles in latest_news.items():
    st.write(f"### {company}")
    for article in articles:
        st.write(f"- [{article['title']}]({article['url']})")

# Input box for manual prediction
st.write("## Manual Prediction")
manual_news = st.text_area("Enter news articles")
if st.button("Predict"):
    manual_preprocessed = preprocess_text(manual_news)
    manual_predictions = predict_stock_prices({'Manual': manual_preprocessed})
    st.write(f"Predicted Price: {manual_predictions['Manual']:.2f}")

# Display stock price charts
st.write("## Stock Price Charts")
for company, ticker in companies.items():
    data = yf.download(ticker, period='1y')
    st.write(f"### {company}")
    st.line_chart(data['Close'])
