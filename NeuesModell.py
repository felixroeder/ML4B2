import pandas as pd
import numpy as np
import spacy
import yfinance as yf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import KNNImputer
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
import nltk
from ta import add_all_ta_features
from textblob import TextBlob
import re
from functools import partial

# Load new financial news dataset
news_data = pd.read_csv('Datensatz.csv')  # Replace with your dataset path
news_data['Date'] = pd.to_datetime(news_data['Date'])
news_data.rename(columns={'News Article': 'News_Article', 'Date': 'Date'}, inplace=True)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Spacy model and NLTK components
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# List of companies to focus on
companies_to_focus = {
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'AAPL': 'Apple'
}

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

# Preprocess news articles
news_data['Processed_Article'] = news_data['News_Article'].apply(preprocess_text)

# Perform Topic Modeling
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(news_data['Processed_Article'])

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda_matrix = lda.fit_transform(tfidf_matrix)

news_data['Topic'] = np.argmax(lda_matrix, axis=1)

# Perform Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

news_data["Sentiment"] = news_data["Processed_Article"].apply(get_sentiment)

# Initialize BERT tokenizer and model (You can also use RoBERTa or other advanced models)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bert_model = TFRobertaModel.from_pretrained('roberta-base')

def get_bert_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token's embedding

# Calculate BERT embeddings for all news
news_data["BERT_Embedding"] = news_data["Processed_Article"].apply(lambda x: get_bert_embeddings([x], tokenizer, bert_model)[0])

# Function to fetch stock prices and fundamental data for each company
def fetch_stock_prices(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.shape[0] > 14:  # Ensure there are at least 15 rows of data
            stock_data = add_all_ta_features(stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume")
            # Handle missing technical indicators
            imputer = KNNImputer(n_neighbors=5)
            stock_data.iloc[:, :] = imputer.fit_transform(stock_data)
        else:
            print(f"Not enough data for {ticker}")
            return pd.DataFrame()

        # Handle missing dates, including weekends and holidays
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        stock_data = stock_data.reindex(all_dates).fillna(method='ffill').fillna(method='bfill').reset_index()
        stock_data.rename(columns={'index': 'Date'}, inplace=True)

        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamentals = stock.info
    return {
        "PE_Ratio": fundamentals.get("trailingPE", np.nan),
        "EPS": fundamentals.get("trailingEps", np.nan),
        "Revenue": fundamentals.get("totalRevenue", np.nan),
        "Market_Cap": fundamentals.get("marketCap", np.nan)
    }

# Correct date format and optionally extend the date range
from_date = "2021-01-01"
to_date = "2021-12-31"  # Extended date range

# Define look-back window
look_back = 5

# Function to prepare data for each company
def prepare_company_data(ticker, company, from_date, to_date):
    print(f"Fetching data for {company} ({ticker})")
    stock_data = fetch_stock_prices(ticker, from_date, to_date)
    if stock_data.empty:
        print(f"No stock data found for {company} ({ticker})")
        return None
    fundamental_data = fetch_fundamental_data(ticker)

    # Filter news for the company or its ticker symbol
    company_news = news_data[news_data['News_Article'].str.contains(company, case=False) | news_data['News_Article'].str.contains(ticker, case=False)]

    # Aggregate all news by day
    all_news_agg = news_data.groupby('Date').agg({
        'BERT_Embedding': lambda x: np.mean(np.vstack(x), axis=0),
        'Sentiment': 'mean'
    }).reset_index()

    # Handle missing dates for all news
    all_dates = pd.date_range(start=from_date, end=to_date, freq='D')
    all_news_agg = all_news_agg.set_index('Date').reindex(all_dates).fillna(method='ffill').fillna(method='bfill').reset_index()
    all_news_agg.rename(columns={'index': 'Date'}, inplace=True)

    # Aggregate company-specific news by day
    if not company_news.empty:
        company_news_agg = company_news.groupby('Date').agg({
            'BERT_Embedding': lambda x: np.mean(np.vstack(x), axis=0),
            'Sentiment': 'mean'
        }).reset_index()

        # Handle missing dates for company-specific news
        company_news_agg = company_news_agg.set_index('Date').reindex(all_dates).fillna(method='ffill').fillna(method='bfill').reset_index()
        company_news_agg.rename(columns={'index': 'Date'}, inplace=True)
    else:
        # Create empty DataFrame with the same structure
        company_news_agg = pd.DataFrame({
            'Date': all_dates,
            'BERT_Embedding': [np.zeros(bert_model.config.hidden_size)] * len(all_dates),
            'Sentiment': [0.0] * len(all_dates)
        })

    # Ensure the columns have correct suffixes
    company_news_agg.rename(columns={'BERT_Embedding': 'BERT_Embedding_company', 'Sentiment': 'Sentiment_company'}, inplace=True)
    all_news_agg.rename(columns={'BERT_Embedding': 'BERT_Embedding_all', 'Sentiment': 'Sentiment_all'}, inplace=True)

    # Merge stock data with aggregated news data
    data = pd.merge(stock_data, company_news_agg, on="Date", how="left")
    data = pd.merge(data, all_news_agg, on="Date", how="left")

    # Add fundamental data (same value for all rows as an example)
    for key, value in fundamental_data.items():
        data[key] = value

    data["Company_Name"] = company

    # Add future price column
    data["Future_Price"] = data["Close"].shift(-1)  # Shift price for prediction

    # Impute missing values in the Future_Price column
    data["Future_Price"].fillna(method='ffill', inplace=True)  # Forward fill
    data["Future_Price"].fillna(method='bfill', inplace=True)  # Backward fill

    # Impute missing values in technical indicators and fundamentals
    technical_indicator_columns = data.filter(like='ta_').columns
    for column in technical_indicator_columns:
        data[column].fillna(method='ffill', inplace=True)
        data[column].fillna(method='bfill', inplace=True)

    fundamental_columns = ["PE_Ratio", "EPS", "Revenue", "Market_Cap"]
    for column in fundamental_columns:
        data[column].fillna(method='ffill', inplace=True)
        data[column].fillna(method='bfill', inplace=True)

    return data

# Prepare data for each company
all_company_data = {ticker: prepare_company_data(ticker, company, from_date, to_date) for ticker, company in companies_to_focus.items()}

# Check for and remove any None entries
all_company_data = {ticker: data for ticker, data in all_company_data.items() if data is not None}

if not all_company_data:
    raise ValueError("No data available for any company in the specified date range.")

# Create sequences for each company
def create_sequences(data, look_back):
    sequences = []
    targets = []
    for i in range(len(data) - look_back):
        sequence = {
            "news_embeddings_company": np.stack(data["BERT_Embedding_company"].values[i:i+look_back]),
            "news_embeddings_all": np.stack(data["BERT_Embedding_all"].values[i:i+look_back]),
            "price": data["Close"].values[i:i+look_back].reshape(-1, 1),
            "sentiment_company": data["Sentiment_company"].values[i:i+look_back].reshape(-1, 1),
            "sentiment_all": data["Sentiment_all"].values[i:i+look_back].reshape(-1, 1),
            "technical_indicators": data.filter(like='ta_').values[i:i+look_back],
            "fundamentals": data[["PE_Ratio", "EPS", "Revenue", "Market_Cap"]].values[i:i+look_back]
        }
        sequences.append(sequence)
        targets.append(data["Future_Price"].values[i + look_back])  # Correctly assign the future price as target
    return sequences, np.array(targets)

company_sequences = {ticker: create_sequences(data, look_back) for ticker, data in all_company_data.items()}

# Ensure consistency of lengths
min_length = min(len(sequences) for sequences, _ in company_sequences.values())
company_sequences = {ticker: (sequences[:min_length], targets[:min_length]) for ticker, (sequences, targets) in company_sequences.items()}

# Convert sequences to arrays for model input
def convert_sequences(sequences):
    news_embeddings_company = np.array([seq["news_embeddings_company"] for seq in sequences])
    news_embeddings_all = np.array([seq["news_embeddings_all"] for seq in sequences])
    price = np.array([seq["price"] for seq in sequences])
    sentiment_company = np.array([seq["sentiment_company"] for seq in sequences])
    sentiment_all = np.array([seq["sentiment_all"] for seq in sequences])
    technical_indicators = np.array([seq["technical_indicators"] for seq in sequences])
    fundamentals = np.array([seq["fundamentals"] for seq in sequences])
    return news_embeddings_company, news_embeddings_all, price, sentiment_company, sentiment_all, technical_indicators, fundamentals

company_features = {ticker: (convert_sequences(sequences), targets) for ticker, (sequences, targets) in company_sequences.items()}

# Validate lengths of the features
for key, (value, targets) in company_features.items():
    print(f"{key} lengths: {[len(x) for x in value]}, targets length: {len(targets)}")

# Combine all features into a single array for RandomizedSearchCV
def combine_features(features):
    combined = np.concatenate([features[0],
                               features[1],
                               features[2],
                               features[3],
                               features[4],
                               features[5],
                               features[6]], axis=-1)
    return combined

combined_features = {ticker: combine_features(features) for ticker, (features, _) in company_features.items()}
combined_features_array = np.concatenate(list(combined_features.values()), axis=0)

# Concatenate all targets into a single array along the correct axis
targets_array = np.concatenate([targets.reshape(-1, 1) for _, targets in company_features.values()], axis=0)

# Ensure the shape of targets matches the expected dimensions
targets_array = targets_array.reshape(-1, len(companies_to_focus))

# Convert targets to a DataFrame for multi-output regression
targets_df = pd.DataFrame(targets_array, columns=companies_to_focus.keys())

# Define the model
def build_model(look_back, combined_dim, num_companies, num_heads=12, ff_dim=128, dropout_rate=0.5):
    combined_input = tf.keras.layers.Input(shape=(look_back, combined_dim), name='combined_input')

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
            self.layernorm2 = tf.keras.kayers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    transformer_block = TransformerBlock(combined_dim, num_heads, ff_dim, rate=dropout_rate)
    x = transformer_block(combined_input)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layer with Batch Normalization and Dropout
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layers for each company
    outputs = {ticker: tf.keras.layers.Dense(1, activation='linear', name=f'output_{ticker}')(x) for ticker in companies_to_focus.keys()}

    # Create model
    model = tf.keras.models.Model(inputs=combined_input, outputs=outputs)

    # Compile model with a dictionary of losses
    losses = {ticker: 'mse' for ticker in companies_to_focus.keys()}
    model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam())

    return model

# Wrap the model with KerasRegressor for use in scikit-learn
def create_keras_model(look_back, combined_dim, num_companies, num_heads=12, ff_dim=128, dropout_rate=0.5):
    return build_model(look_back, combined_dim, num_companies, num_heads, ff_dim, dropout_rate)

# Wrap the model
combined_dim = combined_features_array.shape[-1]
model = KerasRegressor(model=create_keras_model, look_back=look_back, combined_dim=combined_dim,
                       num_companies=len(companies_to_focus), epochs=10, batch_size=32, verbose=1)

# Hyperparameter space
param_distributions = {
    'num_heads': [4, 8, 12],
    'ff_dim': [64, 128, 256],
    'dropout_rate': [0.2, 0.5, 0.7]
}

# Ensure the number of samples is the same
if combined_features_array.shape[0] != targets_df.shape[0]:
    min_samples = min(combined_features_array.shape[0], targets_df.shape[0])
    combined_features_array = combined_features_array[:min_samples]
    targets_df = targets_df.iloc[:min_samples]

# Use functools.partial to set up the model creation with these parameters
model = KerasRegressor(
    build_fn=partial(create_keras_model, look_back=look_back, combined_dim=combined_dim, num_companies=len(companies_to_focus)),
    epochs=10,
    batch_size=32,
    verbose=1
)

# RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=3, verbose=1)

# Prepare combined inputs for training
X_train, X_val, y_train, y_val = train_test_split(combined_features_array, targets_df, test_size=0.2, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best parameters: {best_params}")

# Train the final model with the best parameters
final_model = create_keras_model(look_back, combined_dim, len(companies_to_focus), best_params['num_heads'],
                                 best_params['ff_dim'], best_params['dropout_rate'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the final model
final_model.fit(X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping])

# Make predictions on validation data
predicted_prices = final_model.predict(X_val)

# Convert predictions to a DataFrame for easier handling
predicted_prices_df = pd.DataFrame(predicted_prices, columns=targets_df.columns)

# Display the predicted prices
print(predicted_prices_df.head())

# Save the retrained model
final_model.save('trained_model.keras')
final_model.save('trained_model.h5')
