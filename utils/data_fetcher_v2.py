# TradingRobotTeamV2/utils/data_fetcher_v2.py
"""
Utility functions for fetching financial data using yfinance and NewsAPI.
"""

import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import logging

# Import config variables
from config import (
    PRICE_BOT_MIN_TRAINING_DAYS, # Min days needed for PriceBot
    MAX_NEWS_ARTICLES,           # Max headlines from NewsAPI
    NEWSAPI_KEY,                 # NewsAPI credentials
    NEWSAPI_BASE_URL,
    YF_HISTORY_PERIOD_TRAINING   # yfinance period string for training data
)

log = logging.getLogger(__name__)

# --- Price Data Fetching (Using yfinance) ---

def get_yf_ticker(symbol: str) -> Optional[yf.Ticker]:
    """
    Helper function to create a yfinance Ticker object for a given symbol.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        Optional[yf.Ticker]: The yfinance Ticker object, or None if creation fails.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Basic check: Ticker objects have a 'ticker' attribute equal to the symbol
        if ticker.ticker == symbol.upper():
             return ticker
        else:
             # This case might indicate an issue with the symbol or yfinance internal mapping
             log.warning(f"yfinance Ticker object created for '{symbol}' has unexpected ticker attribute: '{ticker.ticker}'")
             return None # Treat as potentially invalid
    except Exception as e:
        # Catch potential errors during Ticker object creation (e.g., network issues, invalid symbol format)
        log.error(f"Error creating yfinance ticker for '{symbol}': {e}", exc_info=True)
        return None

def get_historical_prices_for_training(symbol: str, period: str = YF_HISTORY_PERIOD_TRAINING, days_needed: int = PRICE_BOT_MIN_TRAINING_DAYS) -> Optional[pd.Series]:
    """
    Fetches historical daily closing prices using yfinance, suitable for training PriceBot.

    Args:
        symbol (str): The stock ticker symbol.
        period (str): The yfinance period string (e.g., "1y", "6mo").
        days_needed (int): Minimum number of trading days data required.

    Returns:
        Optional[pd.Series]: A pandas Series of closing prices (index=datetime, oldest first),
                              or None if fetching fails or insufficient data is returned.
    """
    ticker = get_yf_ticker(symbol)
    if not ticker:
        log.error(f"Cannot fetch historical prices for {symbol}: Invalid yfinance ticker.")
        return None

    try:
        log.info(f"Fetching historical data for {symbol} using yfinance (period={period}). Target days: {days_needed}")
        # Fetch daily data ('1d') for the specified period
        hist = ticker.history(period=period, interval="1d", auto_adjust=True, actions=False) # auto_adjust handles splits/dividends

        if hist.empty:
            log.warning(f"yfinance returned empty history for {symbol} (period={period}).")
            return None

        # Select 'Close' prices, ensure index is datetime, sort oldest first (default)
        prices_series = hist['Close'].sort_index(ascending=True)

        # Check if we received enough data points after fetching
        if len(prices_series) >= days_needed:
            log.info(f"Got {len(prices_series)} historical prices via yfinance for {symbol} training.")
            return prices_series
        else:
            log.warning(f"Insufficient historical data from yfinance for {symbol} training. Got {len(prices_series)}, need {days_needed}.")
            return None
    except Exception as e:
        # Catch potential errors during yfinance download or processing
        log.error(f"Error fetching/processing yfinance history for {symbol}: {e}", exc_info=True)
        return None

def get_recent_prices_for_prediction(symbol: str, days_needed: int) -> Optional[pd.Series]:
    """
    Fetches recent daily closing prices using yfinance for PriceBot prediction.
    Fetches a fixed period ("4mo") to increase chances of getting enough data,
    then returns the latest 'days_needed' points.

    Args:
        symbol (str): The stock ticker symbol.
        days_needed (int): The exact number of recent trading days data required by the bot.

    Returns:
        Optional[pd.Series]: A pandas Series of the N most recent closing prices
                              (index=datetime, oldest first), or None if insufficient data.
    """
    ticker = get_yf_ticker(symbol)
    if not ticker:
        log.error(f"Cannot fetch recent prices for {symbol}: Invalid yfinance ticker.")
        return None

    # Fetch a fixed, reasonably long period to maximize chance of getting enough data points
    fetch_period = "4mo"
    try:
        log.info(f"Fetching recent data for {symbol} using yfinance (fixed period={fetch_period}). Target days needed: {days_needed}")
        hist = ticker.history(period=fetch_period, interval="1d", auto_adjust=True, actions=False)

        if hist.empty:
            log.warning(f"yfinance returned empty history for {symbol} (period={fetch_period}).")
            return None

        prices_series = hist['Close'].sort_index(ascending=True)

        # Check if we got enough data AFTER fetching
        if len(prices_series) >= days_needed:
            # Return only the required number of most recent days using tail()
            recent_prices = prices_series.tail(days_needed)
            log.info(f"Got {len(recent_prices)} most recent daily prices via yfinance for {symbol} prediction (fetched {len(prices_series)} total).")
            return recent_prices
        else:
            # Log if not enough data was available even after fetching the fixed period
            log.warning(f"Could not retrieve enough recent prices via yfinance for {symbol}. Got {len(prices_series)}, need {days_needed}.")
            return None # Return None if the exact required number isn't available
    except Exception as e:
        log.error(f"Error fetching/processing yfinance recent history for {symbol}: {e}", exc_info=True)
        return None


def get_latest_quote_yf(symbol: str) -> Optional[float]:
    """
    Gets the most recent price available from yfinance (likely delayed).
    Prioritizes the last available closing price from daily history.

    Args:
        symbol (str): The stock ticker symbol.

    Returns:
        Optional[float]: The latest available price as a float, or None if unavailable.
    """
    ticker = get_yf_ticker(symbol)
    if not ticker: return None
    try:
        # Fetch data for the last 2 trading days to get the most recent close
        hist = ticker.history(period="2d", interval="1d", auto_adjust=True, actions=False)
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1] # Get the closing price from the last row
            log.debug(f"Latest quote (daily close) for {symbol} via yfinance: {latest_price}")
            # Ensure the price is a valid number before returning
            return float(latest_price) if pd.notna(latest_price) else None
        else:
             # If history fails, maybe log a warning but don't try 'info' as it's less reliable/slower
             log.warning(f"Could not get 2d history for latest quote for {symbol}.")
             return None
    except Exception as e:
        # Log error but don't show full traceback for quote checks unless debugging
        log.error(f"Error fetching latest yfinance quote for {symbol}: {e}", exc_info=log.isEnabledFor(logging.DEBUG))
        return None


# --- News Data Fetching (Using NewsAPI) ---

def fetch_news_headlines(symbol: str, max_headlines: int = MAX_NEWS_ARTICLES) -> List[str]:
    """
    Fetches news headlines for a symbol using the NewsAPI.org service.

    Args:
        symbol (str): The stock symbol or company name to search for.
        max_headlines (int): The maximum number of headlines to return.

    Returns:
        List[str]: A list of news headlines, or an empty list if an error occurs,
                   the API key is missing, or no relevant articles are found.
    """
    if not NEWSAPI_KEY:
        log.warning("NewsAPI key not configured. Cannot fetch news.")
        return []
    if not NEWSAPI_BASE_URL:
         log.error("NEWSAPI_BASE_URL is not configured.")
         return []

    # Parameters for the NewsAPI request
    params = {
        'q': symbol,          # Search query term
        'language': 'en',     # Language of articles
        'sortBy': 'relevancy',# Sort order (relevancy, publishedAt, popularity)
        'pageSize': max_headlines, # Max results per page (and total for free tier often)
        'apiKey': NEWSAPI_KEY # Your API key
    }

    try:
        log.info(f"Fetching news for '{symbol}' from NewsAPI ({NEWSAPI_BASE_URL})")
        # Make the GET request to the NewsAPI endpoint
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10) # 10-second timeout
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        data = response.json() # Parse the JSON response

        # Check the status field in the API response
        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            # Extract the 'title' from each article dictionary, if the title exists
            headlines = [article.get('title') for article in articles if article.get('title')]

            if headlines:
                log.info(f"Fetched {len(headlines)} headlines for '{symbol}' from NewsAPI.")
            else:
                log.info(f"No relevant headlines found for '{symbol}' in NewsAPI response.")
            return headlines
        else:
            # Log error details reported directly by the NewsAPI
            api_error_code = data.get('code', 'UnknownCode')
            api_error_message = data.get('message', 'Unknown error from NewsAPI.')
            log.error(f"NewsAPI error for '{symbol}': [{api_error_code}] {api_error_message}")
            # Provide specific feedback if the API key is the issue
            if api_error_code == 'apiKeyInvalid':
                 log.error("Your NewsAPI key is invalid or missing. Please check config.py / .env file.")
            return []

    except requests.exceptions.Timeout:
         log.error(f"Timeout error fetching news from NewsAPI for {symbol}.")
         return []
    except requests.exceptions.RequestException as e:
        # Catch general request errors (network issues, DNS errors, etc.)
        log.error(f"Error during NewsAPI request for {symbol}: {e}")
        return []
    except Exception as e:
        # Catch other potential errors (e.g., JSON decoding errors)
        log.error(f"Unexpected error processing NewsAPI response for {symbol}: {e}", exc_info=True)
        return []

# Example Usage (for testing when running this file directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Use INFO for direct run

    test_symbol = "AAPL" # Example stock

    print(f"\n--- Testing yfinance Historical Prices ({test_symbol}) ---")
    hist_s_yf = get_historical_prices_for_training(test_symbol)
    if hist_s_yf is not None:
        print(f"Historical Series length: {len(hist_s_yf)}, Last 5 prices:\n{hist_s_yf.tail()}")
    else: print("Failed.")

    print(f"\n--- Testing yfinance Recent Prices ({test_symbol}) ---")
    from config import PRICE_BOT_SMA_PERIOD, PRICE_BOT_INPUT_DAYS
    # Calculate minimum days needed for prediction features
    days_for_pred_test = PRICE_BOT_SMA_PERIOD + PRICE_BOT_INPUT_DAYS
    recent_s_yf = get_recent_prices_for_prediction(test_symbol, days_needed=days_for_pred_test)
    if recent_s_yf is not None:
        print(f"Recent Series length: {len(recent_s_yf)}, Last 5 prices:\n{recent_s_yf.tail()}")
    else: print("Failed.")

    print(f"\n--- Testing yfinance Latest Quote ({test_symbol}) ---")
    latest_q_yf = get_latest_quote_yf(test_symbol)
    if latest_q_yf is not None: print(f"Latest Quote: {latest_q_yf}")
    else: print("Failed.")

    print(f"\n--- Testing News Fetching ({test_symbol}) ---")
    news_h = fetch_news_headlines(test_symbol, max_headlines=3)
    if news_h:
        for i, h in enumerate(news_h): print(f"Headline {i+1}: {h}")
    else: print("Failed to fetch news headlines or none found.")










