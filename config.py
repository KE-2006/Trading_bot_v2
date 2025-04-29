# TradingRobotTeamV2/config.py
import os
from dotenv import load_dotenv
import logging

# --- Environment Setup ---
load_dotenv() # Load secrets from .env file

# --- Logging Configuration ---
# *** CHANGED TO DEBUG TO SEE MORE MESSAGES ***
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
# Get a logger instance to use in other files
log = logging.getLogger(__name__)

# --- API Keys (Optional Alpaca, Required NewsAPI) ---
# Alpaca keys are no longer strictly required for yfinance version
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    log.warning("Alpaca API Key/Secret Key not found. Alpaca functions disabled.")
    # Don't raise error, allow app to run without Alpaca

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    log.warning("NewsAPI Key not found in .env file. News fetching will be disabled.")
    # Don't crash, NewsBot can default to neutral

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# --- Trading Configuration ---
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"] # Stocks to track
DEFAULT_TRADE_QTY = 1 # How many fake shares to trade at once

# --- Simulation Configuration ---
SIM_STARTING_CASH = 10000.00 # Starting fake money

# --- Data Fetching & Bot Configuration ---
PRICE_BOT_INPUT_DAYS = 10
PRICE_BOT_SMA_PERIOD = 20
PRICE_BOT_MIN_TRAINING_DAYS = PRICE_BOT_SMA_PERIOD + 5
MAX_NEWS_ARTICLES = 5
# How often (in seconds) to fetch "live-ish" prices using yfinance
LIVE_PRICE_POLL_INTERVAL = 60 # Fetch every 60 seconds

# yfinance period strings for fetching data
# See yfinance docs for options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
YF_HISTORY_PERIOD_TRAINING = "1y" # Fetch 1 year for training
YF_HISTORY_PERIOD_PREDICTION = "3mo" # Fetch 3 months for prediction (ensure enough days)

# --- File Paths ---
TRADE_LOG_FILE = "trade_log_yf_sim.txt" # Use a different log file name

log.info("Configuration loaded successfully (yfinance version).")
log.info(f"Trading Symbols: {SYMBOLS}")
log.info(f"PriceBot Input Days: {PRICE_BOT_INPUT_DAYS}, SMA Period: {PRICE_BOT_SMA_PERIOD}")
log.info(f"NewsAPI Key Loaded: {'Yes' if NEWSAPI_KEY else 'No'}")
log.info(f"Simulation Starting Cash: ${SIM_STARTING_CASH:.2f}")
# Log the effective log level
log.info(f"Logging level set to: {logging.getLevelName(log.getEffectiveLevel())}")


