import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

def get_news_sentiment(stock_tickers):
    sentiment_dict = {}
    
    # Step 1: Iterate over each stock ticker
    for ticker in stock_tickers:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        print(f"Fetching data for {ticker} from {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Step 2: Extract headlines related to the stock
        headlines = soup.find_all('a', class_='tab-link-news')

        if not headlines:
            print(f"No headlines found for {ticker}")
            continue
        
        # Step 3: Perform sentiment analysis on the headlines
        sentiment_score = 0
        for headline in headlines:
            text = headline.get_text()
            analysis = TextBlob(text)
            sentiment_score += analysis.sentiment.polarity
        
        # Step 4: Calculate average sentiment and store in dictionary
        average_sentiment = sentiment_score / len(headlines)
        sentiment_dict[ticker] = average_sentiment
    
    return sentiment_dict

# Example usage:
stock_tickers = ['AAPL', 'GOOGL', 'TSLA']
sentiment_analysis = get_news_sentiment(stock_tickers)

print(sentiment_analysis)
