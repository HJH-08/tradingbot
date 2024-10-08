from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    # Remove unwanted characters and normalize the text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    return text

def get_news_sentiment_selenium(stock_tickers):
    sentiment_dict = {}
    
    # Initialize the Selenium WebDriver
    driver = webdriver.Chrome()  # Ensure chromedriver is in your PATH

    for ticker in stock_tickers:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        driver.get(url)
        
        try:
            # Wait until the news headlines are present on the page
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'tab-link-news'))
            )
            
            # Parse the page source after it fully loads
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            headlines = soup.find_all('a', class_='tab-link-news')
            
            if not headlines:
                print(f"No headlines found for {ticker}")
                continue
            
            sentiment_score = 0
            for headline in headlines:
                text = headline.get_text()
                text = preprocess_text(text)
                
                # Use VADER for sentiment analysis
                analysis = analyzer.polarity_scores(text)
                sentiment_score += analysis['compound']
            
            average_sentiment = sentiment_score / len(headlines)
            sentiment_dict[ticker] = average_sentiment
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    driver.quit()
    return sentiment_dict

# Example usage:
stock_tickers = ['AAPL', 'GOOGL', 'TSLA']
sentiment_analysis = get_news_sentiment_selenium(stock_tickers)

print(sentiment_analysis)
