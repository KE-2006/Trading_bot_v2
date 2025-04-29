# TradingRobotTeamV2/utils/data_fetcher_v2.py
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import logging

# Import config variables
from config import (
    PRICE_BOT_MIN_TRAINING_DAYS,
    MAX_NEWS_ARTICLES,
    NEWSAPI_KEY,
    NEWSAPI_BASE_URL,
    YF_HISTORY_PERIOD_TRAINING,
    # YF_HISTORY_PERIOD_PREDICTION # No longer using this directly here
)

log = logging.getLogger(__name__)

# --- Price Data Fetching (Using yfinance) ---

def get_yf_ticker(symbol: str) -> Optional[yf.Ticker]:
    """Helper function to get a yfinance Ticker object."""
    try:
        ticker = yf.Ticker(symbol)
        return ticker
    except Exception as e:
        log.error(f"Error creating yfinance ticker for {symbol}: {e}", exc_info=True)
        return None

def get_historical_prices_for_training(symbol: str, period: str = YF_HISTORY_PERIOD_TRAINING, days_needed: int = PRICE_BOT_MIN_TRAINING_DAYS) -> Optional[pd.Series]:
    """Fetches historical daily closing prices using yfinance for training."""
    ticker = get_yf_ticker(symbol)
    if not ticker: return None
    try:
        log.info(f"Fetching historical data for {symbol} using yfinance (period={period}). Target days: {days_needed}")
        hist = ticker.history(period=period, interval="1d")
        if hist.empty:
            log.warning(f"yfinance returned empty history for {symbol} (period={period}).")
            return None
        prices_series = hist['Close'].sort_index(ascending=True)
        if len(prices_series) >= days_needed:
            log.info(f"Got {len(prices_series)} historical prices via yfinance for {symbol} training.")
            return prices_series
        else:
            log.warning(f"Insufficient historical data from yfinance for {symbol} training. Got {len(prices_series)}, need {days_needed}.")
            return None
    except Exception as e:
        log.error(f"Error fetching/processing yfinance history for {symbol}: {e}", exc_info=True)
        return None

def get_recent_prices_for_prediction(symbol: str, days_needed: int) -> Optional[pd.Series]:
    """
    Fetches recent daily closing prices using yfinance for prediction.
    Fetches a fixed period ("4mo") and checks if enough data exists.
    """
    ticker = get_yf_ticker(symbol)
    if not ticker: return None

    # *** MODIFICATION: Fetch fixed period, then check length ***
    fetch_period = "4mo" # Fetch a slightly longer fixed period
    try:
        log.info(f"Fetching recent data for {symbol} using yfinance (fixed period={fetch_period}). Target days needed: {days_needed}")
        hist = ticker.history(period=fetch_period, interval="1d")

        if hist.empty:
            log.warning(f"yfinance returned empty history for {symbol} (period={fetch_period}).")
            return None

        prices_series = hist['Close'].sort_index(ascending=True)

        # *** Check if we got enough data AFTER fetching ***
        if len(prices_series) >= days_needed:
            # Return only the required number of most recent days
            recent_prices = prices_series.tail(days_needed)
            log.info(f"Got {len(recent_prices)} most recent daily prices via yfinance for {symbol} prediction (fetched {len(prices_series)} total).")
            return recent_prices
        else:
            log.warning(f"Could not retrieve enough recent prices via yfinance for {symbol}. Got {len(prices_series)}, need {days_needed}.")
            return None # Return None if not enough data
    except Exception as e:
        log.error(f"Error fetching/processing yfinance recent history for {symbol}: {e}", exc_info=True)
        return None
    # *** End of Modification ***


def get_latest_quote_yf(symbol: str) -> Optional[float]:
    """Gets the most recent price available from yfinance (likely delayed)."""
    ticker = get_yf_ticker(symbol)
    if not ticker: return None
    try:
        hist = ticker.history(period="2d", interval="1d")
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            log.debug(f"Latest quote (daily close) for {symbol} via yfinance: {latest_price}")
            return float(latest_price) if pd.notna(latest_price) else None
        else:
             log.warning(f"Could not get latest quote for {symbol} from yfinance history.")
             return None
    except Exception as e:
        log.error(f"Error fetching latest yfinance quote for {symbol}: {e}", exc_info=False)
        return None


# --- News Data Fetching (Using NewsAPI - Keep as is) ---
def fetch_news_headlines(symbol: str, max_headlines: int = MAX_NEWS_ARTICLES) -> List[str]:
    if not NEWSAPI_KEY:
        log.warning("NewsAPI key not configured. Cannot fetch news.")
        return []
    if not NEWSAPI_BASE_URL:
         log.error("NEWSAPI_BASE_URL is not configured.")
         return []
    params = { 'q': symbol, 'language': 'en', 'sortBy': 'relevancy', 'pageSize': max_headlines, 'apiKey': NEWSAPI_KEY }
    try:
        log.info(f"Fetching news for '{symbol}' from NewsAPI ({NEWSAPI_BASE_URL})")
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            headlines = [article.get('title') for article in articles if article.get('title')]
            if headlines: log.info(f"Fetched {len(headlines)} headlines for '{symbol}' from NewsAPI.")
            else: log.info(f"No relevant headlines found for '{symbol}' in NewsAPI response.")
            return headlines
        else:
            api_error_code = data.get('code', 'UnknownCode')
            api_error_message = data.get('message', 'Unknown error from NewsAPI.')
            log.error(f"NewsAPI error for '{symbol}': [{api_error_code}] {api_error_message}")
            if api_error_code == 'apiKeyInvalid': log.error("Your NewsAPI key is invalid. Please check config/.env file.")
            return []
    except requests.exceptions.Timeout:
         log.error(f"Timeout error fetching news from NewsAPI for {symbol}.")
         return []
    except requests.exceptions.RequestException as e:
        log.error(f"Error during NewsAPI request for {symbol}: {e}")
        return []
    except Exception as e:
        log.error(f"Unexpected error processing NewsAPI response for {symbol}: {e}", exc_info=True)
        return []

# Example Usage (Keep as is for testing)
# ... (rest of the file remains the same) ...
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Use INFO for direct run

    test_symbol = "AAPL"

    print(f"\n--- Testing yfinance Historical Prices ({test_symbol}) ---")
    hist_s_yf = get_historical_prices_for_training(test_symbol)
    if hist_s_yf is not None:
        print(f"Historical Series length: {len(hist_s_yf)}, Last 5 prices:\n{hist_s_yf.tail()}")
    else: print("Failed.")

    print(f"\n--- Testing yfinance Recent Prices ({test_symbol}) ---")
    from config import PRICE_BOT_SMA_PERIOD, PRICE_BOT_INPUT_DAYS
    days_for_pred = PRICE_BOT_SMA_PERIOD + PRICE_BOT_INPUT_DAYS # This is min_data_length
    recent_s_yf = get_recent_prices_for_prediction(test_symbol, days_needed=days_for_pred)
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









