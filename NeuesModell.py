import os
import re
import pandas as pd
import numpy as np
import spacy
import requests
import yfinance as yf
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, KFold
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Input, Concatenate, LayerNormalization, Dropout, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import nltk
import matplotlib.pyplot as plt
import streamlit as st
import newsapi
from datetime import datetime, timedelta

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Spacy model and NLTK components
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Load dataset
data = pd.read_csv('Datensatz.csv')

# Column names
date_col = "Date"
news_col = "News Article"
price_cols = ["Apple_Price", "Amazon_Price", "Google_Price"]
companies = ["Apple", "Amazon", "Google"]

# Convert date column to datetime
data[date_col] = pd.to_datetime(data[date_col])
data.sort_values(by=date_col, inplace=True)

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
data[news_col] = data[news_col].apply(preprocess_text)

# Perform NER
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
    return entities

data["Entities"] = data[news_col].apply(extract_entities)

# Function to check if a news article is relevant to a company
def is_relevant(entities, company):
    return int(company.lower() in (e.lower() for e in entities))

# Create relevance features for each company
for company in companies:
    data[f"{company}_Relevant"] = data["Entities"].apply(lambda x: is_relevant(x, company))

# Perform Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

data["Sentiment"] = data[news_col].apply(get_sentiment)

# Apply TF-IDF Vectorization
vectorizer_tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer_tfidf.fit_transform(data[news_col]).toarray()

# Apply Topic Modeling with LDA
n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
topics_matrix = lda.fit_transform(tfidf_matrix)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = bert_model(inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
    return embeddings

# Get BERT embeddings for news articles
bert_embeddings = get_bert_embeddings(data[news_col].tolist())

# Combine features
data["TFIDF"] = list(tfidf_matrix)
data["Topics"] = list(topics_matrix)
data["BERT_Embeddings"] = list(bert_embeddings)

data.head()

# Create a set of all unique entity types in the dataset
unique_entities = set(entity for entities in data["Entities"] for entity in entities)
entity_to_index = {entity: idx for idx, entity in enumerate(unique_entities)}

def create_entity_features(entities, entity_to_index):
    feature_vector = np.zeros(len(entity_to_index))
    for entity in entities:
        if entity in entity_to_index:
            feature_vector[entity_to_index[entity]] = 1
    return feature_vector

data["Entity_Features"] = data["Entities"].apply(lambda x: create_entity_features(x, entity_to_index))

# Encode company names as indices
company_encoder = LabelEncoder()
data["Company_Indices"] = company_encoder.fit_transform(data[price_cols].idxmax(axis=1))

# Define look-back window
look_back = 5

def create_sequences(features, price_col, company_idx, window_size):
    sequences = []
    for i in range(len(features) - window_size):
        bert_sequence = np.array(features["BERT_Embeddings"].tolist())[i:i+window_size]
        price_sequence = features[price_col].values[i:i+window_size].reshape(-1, 1)
        company_sequence = np.array([company_idx] * window_size)
        entity_sequence = np.array(features["Entity_Features"].tolist())[i:i+window_size]
        sentiment_sequence = features["Sentiment"].values[i:i+window_size].reshape(-1, 1)
        tfidf_sequence = np.array(features["TFIDF"].tolist())[i:i+window_size]
        topics_sequence = np.array(features["Topics"].tolist())[i:i+window_size]
        relevance_sequence = features[f"{companies[company_idx]}_Relevant"].values[i:i+window_size].reshape(-1, 1)
        sequence = {
            "bert": bert_sequence,
            "price": price_sequence,
            "company": company_sequence,
            "entities": entity_sequence,
            "sentiment": sentiment_sequence,
            "tfidf": tfidf_sequence,
            "topics": topics_sequence,
            "relevance": relevance_sequence
        }
        sequences.append(sequence)
    return sequences

# Create sequences for each company
all_sequences = []
for company_idx, price_col in enumerate(price_cols):
    all_sequences += create_sequences(data.copy(), price_col, company_idx, look_back)

# Convert sequences to appropriate format
def convert_sequences(sequences):
    bert = np.array([seq["bert"] for seq in sequences])
    price = np.array([seq["price"] for seq in sequences])
    company = np.array([seq["company"] for seq in sequences])
    entities = np.array([seq["entities"] for seq in sequences])
    sentiment = np.array([seq["sentiment"] for seq in sequences])
    tfidf = np.array([seq["tfidf"] for seq in sequences])
    topics = np.array([seq["topics"] for seq in sequences])
    relevance = np.array([seq["relevance"] for seq in sequences])
    return bert, price, company, entities, sentiment, tfidf, topics, relevance

bert, price, company, entities, sentiment, tfidf, topics, relevance = convert_sequences(all_sequences)

# Use only the first 199 sequences to match the target arrays
bert = bert[:199]
price = price[:199]
company = company[:199]
entities = entities[:199]
sentiment = sentiment[:199]
tfidf = tfidf[:199]
topics = topics[:199]
relevance = relevance[:199]

# Verify shapes of the converted sequences
print(bert.shape)
print(price.shape)
print(company.shape)
print(entities.shape)
print(sentiment.shape)
print(tfidf.shape)
print(topics.shape)
print(relevance.shape)

# Adjust target data to include separate outputs for each company
targets = {f'output_{company}': data[price_col].shift(-1).dropna().values[:len(all_sequences)] for price_col, company in zip(price_cols, companies)}

# Ensure the target data aligns with the sequences
aligned_targets = {}
for key, value in targets.items():
    aligned_value = value[:len(bert)]
    aligned_targets[key] = aligned_value

# Ensure no NaNs in targets
for key in aligned_targets.keys():
    aligned_targets[key] = np.nan_to_num(aligned_targets[key])

# Verify shapes of the aligned targets
for key, value in aligned_targets.items():
    print(f"{key}: {value.shape}")

# Check for NaNs in Input Data
for input_data in [bert, price, company, entities, sentiment, tfidf, topics, relevance]:
    assert not np.isnan(input_data).any(), "Found NaNs in input data"

# Check for NaNs in Targets
for key, value in aligned_targets.items():
    assert not np.isnan(value).any(), f"Found NaNs in target data for {key}"

# Required for reshaping 
class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

# Build Transformer model
def build_transformer_model():
    bert_input = Input(shape=(look_back, bert.shape[2]), name='bert_input')
    price_input = Input(shape=(look_back, 1), name='price_input')
    company_input = Input(shape=(look_back,), name='company_input')
    entities_input = Input(shape=(look_back, len(unique_entities)), name='entities_input')
    sentiment_input = Input(shape=(look_back, 1), name='sentiment_input')
    tfidf_input = Input(shape=(look_back, tfidf.shape[2]), name='tfidf_input')
    topics_input = Input(shape=(look_back, n_topics), name='topics_input')
    relevance_input = Input(shape=(look_back, 1), name='relevance_input')

    # Apply dense layers to each input to ensure consistent dimensions
    bert_dense = TimeDistributed(Dense(128))(bert_input)
    price_dense = TimeDistributed(Dense(128))(price_input)
    company_embedding_layer = Embedding(input_dim=len(price_cols) + 1, output_dim=128, name='company_embedding')
    company_dense = company_embedding_layer(company_input)
    company_dense = ReshapeLayer([-1, look_back, 128])(company_dense)
    entities_dense = TimeDistributed(Dense(128))(entities_input)
    sentiment_dense = TimeDistributed(Dense(128))(sentiment_input)
    tfidf_dense = TimeDistributed(Dense(128))(tfidf_input)
    topics_dense = TimeDistributed(Dense(128))(topics_input)
    relevance_dense = TimeDistributed(Dense(128))(relevance_input)

    # Combine all inputs
    combined = Concatenate(axis=-1)([bert_dense, price_dense, company_dense, entities_dense, sentiment_dense, tfidf_dense, topics_dense, relevance_dense])

    # Transformer block
    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ])
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    embed_dim = combined.shape[-1]  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 1024  # Hidden layer size in feed forward network inside transformer

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

     # Use a Lambda layer to pass the training argument
    x = tf.keras.layers.Lambda(lambda inputs: transformer_block(inputs, training=True))(combined)

    # Regularization
    x = Dropout(0.2)(x)

    # Create separate output layers for each company
    outputs = {f'output_{company}': Dense(1, activation='linear', name=f'output_{company}')(x[:, -1, :]) for company in companies}

    # Create model
    model = Model(inputs=[bert_input, price_input, company_input, entities_input, sentiment_input, tfidf_input, topics_input, relevance_input], outputs=outputs)
    return model

# Prepare inputs for the model
inputs = [bert, price, company, entities, sentiment, tfidf, topics, relevance]

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to compile and train the model within each fold
def compile_and_train_model(train_idx, val_idx):
    train_inputs = [input_data[train_idx] for input_data in inputs]
    val_inputs = [input_data[val_idx] for input_data in inputs]

    train_targets_split = {key: value[train_idx] for key, value in targets.items()}
    val_targets_split = {key: value[val_idx] for key, value in targets.items()}

    # Build the Transformer model
    model = build_transformer_model()
    model.compile(loss={f'output_{company}': 'mse' for company in companies}, optimizer=Adam())

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_inputs,
        train_targets_split,
        validation_data=(val_inputs, val_targets_split),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )

    return model, history

# Perform K-Fold cross-validation
fold_models = []
histories = []
for train_idx, val_idx in kf.split(np.arange(len(bert))):
    model, history = compile_and_train_model(train_idx, val_idx)
    fold_models.append(model)
    histories.append(history)

# Function to make predictions using the average of models from cross-validation
def predict_with_ensemble(models, inputs):
    predictions = [model.predict(inputs) for model in models]
    avg_predictions = {}
    for company in companies:
        avg_predictions[f'output_{company}'] = np.mean([pred[f'output_{company}'] for pred in predictions], axis=0)
    return avg_predictions

# Make predictions on test data with ensemble of models from cross-validation
test_predictions = predict_with_ensemble(fold_models, inputs)

# Convert predictions to a DataFrame for easier handling
predicted_prices_df = pd.DataFrame({company: test_predictions[f'output_{company}'].flatten() for company in companies})

# Display the predicted prices
print(predicted_prices_df.head())

# Save the retrained model
model.save('retrained_model.keras')
