from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import torch

# Initialize the sentiment analysis pipeline using a transformer model
sentiment_model = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

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
            
            sentiment_scores = []
            for headline in headlines:
                text = headline.get_text()
                
                # Perform sentiment analysis using the transformer model
                analysis = sentiment_model(text)
                
                # Extract sentiment score (convert 'POSITIVE'/'NEGATIVE' labels to +1/-1)
                score = 1 if analysis[0]['label'] == 'POSITIVE' else -1
                sentiment_scores.append(score * analysis[0]['score'])
            
            # Calculate the average sentiment score for the ticker
            if sentiment_scores:
                average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_dict[ticker] = average_sentiment
            else:
                sentiment_dict[ticker] = 0
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    driver.quit()
    return sentiment_dict

# Example usage:
stock_tickers = ['AAPL', 'GOOGL', 'TSLA']
sentiment_analysis = get_news_sentiment_selenium(stock_tickers)

print(sentiment_analysis)
