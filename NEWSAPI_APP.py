!pip install streamlit requests

import streamlit as st
import requests

# Function to fetch news
def fetch_news(query):
    api_key = 'ae91264b58784ff19f181a6691c1efc6'
    url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    return response.json()

# Streamlit App
st.title('Latest News from Google, Apple, and Amazon')

# Fetch news for each company
queries = ['Google', 'Apple', 'Amazon']
news_data = {query: fetch_news(query) for query in queries}

# Display news
for query in queries:
    st.header(f'Latest News about {query}')
    articles = news_data[query].get('articles', [])
    
    if not articles:
        st.write(f'No news found for {query}.')
    else:
        for article in articles[:5]:  # Display top 5 articles
            st.subheader(article['title'])
            st.write(article['description'])
            st.write(f"[Read more]({article['url']})")
            st.write(f"Published at: {article['publishedAt']}")
            st.write("---")

# Run Streamlit App
if __name__ == '__main__':
    st.run()


streamlit run news_app.py
