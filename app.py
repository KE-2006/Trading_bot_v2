# TradingRobotTeamV2/app.py (yfinance + Simulator Version - Cleaned)
"""
Main Flask application file for the Trading Robot Dashboard.
Uses yfinance for data, NewsAPI for news, and an internal simulator for trading.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pandas as pd
import datetime
import threading # For background polling thread
import time
import os
import logging
from typing import Dict, Optional, Any, List # <-- IMPORT LIST HERE

# Import project components
from config import (
    SYMBOLS, TRADE_LOG_FILE, NEWSAPI_KEY,
    PRICE_BOT_INPUT_DAYS, PRICE_BOT_SMA_PERIOD, PRICE_BOT_MIN_TRAINING_DAYS,
    LIVE_PRICE_POLL_INTERVAL,
    log # Use logger from config
)
from agents.price_bot_v2 import PriceBotV2
from agents.news_bot import NewsBot
from agents.trade_bot_v2 import TradeBotV2 # Ensure using the correct cleaned version
from utils.data_fetcher_v2 import (
    get_historical_prices_for_training,
    get_recent_prices_for_prediction,
    fetch_news_headlines,
    get_latest_quote_yf
)
from utils.portfolio_simulator import PortfolioSimulator

# --- Flask App Initialization ---
app = Flask(__name__)
# Secret key is required for session management and flash messages
app.secret_key = os.urandom(24) # Generates a random key each time app starts

# --- Global Variables & Initialization ---
# Initialize Portfolio Simulator
simulator = PortfolioSimulator(symbols=SYMBOLS)

# Initialize Bots for each symbol
bots: Dict[str, Dict[str, Any]] = {}
for symbol in SYMBOLS:
    bots[symbol] = {
        "price_bot": PriceBotV2(input_days=PRICE_BOT_INPUT_DAYS, sma_period=PRICE_BOT_SMA_PERIOD),
        "news_bot": NewsBot(),
        "trade_bot": TradeBotV2(safety_on=True)
    }
log.info(f"Initialized V2 bots for symbols: {list(bots.keys())}")

# Dictionary to store the latest polled prices
latest_prices: Dict[str, float] = {symbol: 0.0 for symbol in SYMBOLS}
# Thread lock to prevent race conditions when accessing latest_prices
price_lock = threading.Lock()

# --- Background Polling Thread for Prices ---
def poll_live_prices():
    """
    Runs in a background thread. Periodically fetches the latest available quote
    for each symbol using yfinance and updates the global latest_prices dictionary.
    """
    global latest_prices
    log.info("Starting background thread for polling yfinance quotes...")
    while True:
        log.debug(f"Polling yfinance for latest quotes for: {SYMBOLS}")
        temp_prices = {} # Store results for this poll cycle
        for symbol in SYMBOLS:
            quote = get_latest_quote_yf(symbol)
            if quote is not None:
                temp_prices[symbol] = quote
            else:
                # If fetch fails, log warning and keep the previously known price
                log.warning(f"Polling failed to get quote for {symbol}, using previous value.")
                with price_lock:
                     temp_prices[symbol] = latest_prices.get(symbol, 0.0) # Use get() for safety

            # Add a small delay between API calls to avoid rate limiting
            time.sleep(0.5)

        # Update the shared global dictionary safely using the lock
        with price_lock:
            latest_prices.update(temp_prices)
        log.debug(f"Updated latest_prices: {latest_prices}")

        # Wait for the configured interval before the next poll cycle
        poll_interval_seconds = max(10, LIVE_PRICE_POLL_INTERVAL) # Ensure at least 10 sec interval
        time.sleep(poll_interval_seconds)

# Create and start the background polling thread
# daemon=True ensures the thread exits when the main app exits
polling_thread = threading.Thread(target=poll_live_prices, daemon=True)
polling_thread.start()


# --- Helper Functions ---
# Use the imported List for type hinting here
def read_trade_log() -> List[Dict[str, str]]:
    """
    Reads and parses the trade log file.

    Returns:
        List[Dict[str, str]]: A list of trade log entries (dictionaries), newest first.
                               Returns empty list if file not found or on error.
    """
    log_entries = []
    if not os.path.exists(TRADE_LOG_FILE):
        log.warning(f"Trade log file '{TRADE_LOG_FILE}' not found.")
        return log_entries
    try:
        with open(TRADE_LOG_FILE, "r") as f:
            lines = f.readlines()
            for line in reversed(lines): # Process newest first
                parts = line.strip().split(" | ")
                # Basic check for expected number of parts
                if len(parts) >= 5:
                    # Extract relevant parts, handling potential missing ': '
                    decision_part = parts[3].split(': ')[-1]
                    result_part = parts[4].split(': ')[-1]
                    details_part = " | ".join(parts[5:]) if len(parts) > 5 else ""
                    entry = {
                        "timestamp": parts[0],
                        "symbol": parts[1],
                        "decision": decision_part,
                        "result": result_part,
                        "details": details_part
                    }
                    log_entries.append(entry)
                else:
                    log.warning(f"Skipping malformed log line: {line.strip()}")
    except Exception as e:
        log.error(f"Error reading trade log file '{TRADE_LOG_FILE}': {e}")
    return log_entries

def write_trade_log(log_data: Dict[str, Any]):
    """
    Appends a structured entry to the trade log file based on analysis results.

    Args:
        log_data (Dict[str, Any]): Dictionary containing results from the analysis process.
    """
    try:
        # Construct the log line string with consistent formatting
        log_line = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{log_data.get('symbol', 'N/A')} | "
            f"Prices Used: {len(log_data.get('prices_list', []))} days | "
            f"Decision: {log_data.get('decision', 'N/A')} | "
            f"Result: {log_data.get('trade_result_msg', 'N/A')} | "
            f"Cash: {log_data.get('cash', 'N/A')} | "
            f"Shares ({log_data.get('symbol', 'N/A')}): {log_data.get('shares', 'N/A')} | "
            f"Portfolio: {log_data.get('portfolio', 'N/A')} | "
            f"Profit: {log_data.get('profit', 'N/A')} | "
            # Log the signal values used for the decision
            f"Signals: P={log_data.get('price_signal_text','?')}, "
            f"N={log_data.get('news_signal_text','?')}, "
            f"SMA={'A' if log_data.get('price_above_sma') else 'B' if log_data.get('price_above_sma') is False else '?'}\n"
        )
        # Append the line to the log file
        with open(TRADE_LOG_FILE, "a") as diary:
            diary.write(log_line)
        log.info(f"Trade log entry written for {log_data.get('symbol', 'N/A')}.")
    except Exception as e:
        log.error(f"Error writing to trade log file '{TRADE_LOG_FILE}': {e}", exc_info=True)


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    """
    Handles the main dashboard display (GET) and processes analysis/trade requests (POST).
    """
    analysis_result = None # Initialize analysis result for this request

    # Handle form submission for analysis/trading
    if request.method == 'POST' and request.form.get('action') == 'analyze_trade':
        chosen_symbol = request.form.get('chosen_stock')
        if chosen_symbol in SYMBOLS:
            log.info(f"--- yfinance/Sim Analysis & Trade Action for {chosen_symbol} ---")
            analysis_result = {'symbol': chosen_symbol} # Store symbol

            # --- Step 1: Get Data ---
            # Determine minimum days needed for price bot features/prediction
            days_needed_for_pred = bots[chosen_symbol]["price_bot"].min_data_length
            log.debug(f"Fetching {days_needed_for_pred} recent daily prices for {chosen_symbol}...")
            prices_series = get_recent_prices_for_prediction(chosen_symbol, days_needed=days_needed_for_pred)

            log.debug(f"Fetching news headlines for {chosen_symbol}...")
            news_headlines = fetch_news_headlines(chosen_symbol)
            analysis_result['news'] = news_headlines # Store headlines (might be empty)

            # Store prices as list for display, handle potential None
            analysis_result['prices'] = prices_series.tolist() if prices_series is not None else "Not available"

            # --- Step 1b: Validate Data ---
            if prices_series is None:
                 # If price data fetching failed, log error and skip analysis
                 error_msg = f"Insufficient recent price data for {chosen_symbol} analysis (yfinance)."
                 log.warning(error_msg)
                 analysis_result['error'] = error_msg
                 analysis_result['decision'] = "HOLD" # Cannot make decision
                 analysis_result['trade_result'] = {"success": False, "message": "Analysis failed: Insufficient price data.", "order_id": None, "error": "InsufficientData"}
            else:
                # --- Step 2: Get Bot & SMA Signals ---
                price_bot = bots[chosen_symbol]["price_bot"]
                news_bot = bots[chosen_symbol]["news_bot"]
                trade_bot = bots[chosen_symbol]["trade_bot"]

                # Ensure PriceBot is trained (train only if needed)
                if not price_bot.is_trained:
                    log.info(f"PriceBotV2 for {chosen_symbol} not trained. Training now...")
                    hist_prices_series = get_historical_prices_for_training(chosen_symbol)
                    if hist_prices_series is not None:
                        # Pass list to train method
                        price_bot.train(hist_prices_series.tolist())
                    else:
                         log.warning(f"Could not train PriceBotV2 for {chosen_symbol} - yfinance history fetch failed.")

                # Get prediction from PriceBot (expects list)
                price_signal = price_bot.predict(prices_series.tolist())
                # Get sentiment from NewsBot, default to Neutral (0) on failure or no news
                news_signal_raw = news_bot.analyze(news_headlines)
                news_signal = 0 if news_signal_raw is None or not news_headlines else news_signal_raw

                # Calculate Price vs SMA signal
                price_above_sma = None
                try:
                    log.debug(f"Calculating SMA signal for {chosen_symbol}. Data length: {len(prices_series)}, SMA Period: {PRICE_BOT_SMA_PERIOD}")
                    if len(prices_series) >= PRICE_BOT_SMA_PERIOD:
                         sma_series = prices_series.rolling(window=PRICE_BOT_SMA_PERIOD).mean()
                         current_sma = sma_series.iloc[-1]
                         latest_price = prices_series.iloc[-1]
                         log.debug(f"SMA Calculation: Latest Price = {latest_price}, Calculated SMA = {current_sma}")
                         if pd.notna(current_sma) and pd.notna(latest_price):
                              price_above_sma = latest_price > current_sma
                              log.debug(f"SMA Result: PriceAboveSMA = {price_above_sma}")
                         else:
                             log.warning(f"Could not compare Price vs SMA for {chosen_symbol} (SMA or Price is NaN).")
                    else:
                         log.warning(f"Not enough data ({len(prices_series)}) for SMA({PRICE_BOT_SMA_PERIOD}) for {chosen_symbol}.")
                except Exception as e:
                    log.error(f"Error calculating SMA signal for {chosen_symbol}: {e}", exc_info=True)

                # Store signals in result dictionary for display and logging
                analysis_result['price_signal'] = price_signal
                analysis_result['news_signal'] = news_signal
                analysis_result['price_above_sma'] = price_above_sma
                analysis_result['price_signal_text'] = "Up" if price_signal == 1 else "Down" if price_signal == 0 else "N/A"
                analysis_result['news_signal_text'] = "Good" if news_signal == 1 else "Bad" if news_signal == -1 else "Neutral"
                log.info(f"{chosen_symbol} Signals - Price: {analysis_result['price_signal_text']}, News: {analysis_result['news_signal_text']}, SMA: {'Above' if price_above_sma else 'Below' if price_above_sma is False else 'N/A'}")

                # --- Step 3: Get Trade Decision ---
                # Pass signals and prices list to TradeBot
                decision = trade_bot.decide(price_signal, news_signal, price_above_sma, prices_series.tolist())
                analysis_result['decision'] = decision
                log.info(f"{chosen_symbol} Trade Decision: {decision}")

                # --- Step 4: Execute Trade using Simulator ---
                trade_result_dict = {"success": True, "message": "Decision is HOLD.", "order_id": None, "error": None}
                if decision in ["BUY", "SELL"]:
                    # Get the latest polled price for simulation (use lock for thread safety)
                    with price_lock:
                        sim_price = latest_prices.get(chosen_symbol)
                    # Only simulate if price is valid
                    if sim_price and sim_price > 0:
                        log.info(f"Simulating {decision} for {chosen_symbol} @ price {sim_price:.2f}...")
                        # Call the simulator's trade method
                        trade_result_dict = simulator.simulate_trade(decision, chosen_symbol, 1, sim_price) # Use default qty=1
                    else:
                        # Handle case where latest price isn't available for simulation
                        log.warning(f"Cannot simulate {decision} for {chosen_symbol}: Latest price unknown ({sim_price}).")
                        trade_result_dict = {"success": False, "message": f"Simulation failed: Price unavailable ({sim_price}).", "order_id": None, "error": "PriceUnavailable"}
                    log.info(f"{chosen_symbol} Simulation Result: {trade_result_dict}")
                analysis_result['trade_result'] = trade_result_dict # Store simulation result

            # --- Step 5: Log the action ---
            # Get current simulated portfolio state AFTER potential trade
            with price_lock:
                current_prices_for_log = latest_prices.copy() # Get consistent prices for valuation
            sim_account = simulator.get_account_details(current_prices_for_log)
            sim_position = simulator.get_current_position(chosen_symbol)

            # Prepare data dictionary for the log writer function
            log_data = analysis_result.copy()
            log_data['trade_result_msg'] = analysis_result['trade_result']['message']
            log_data['cash'] = f"${sim_account.get('cash', 'N/A'):.2f}"
            log_data['shares'] = sim_position
            log_data['portfolio'] = f"${sim_account.get('portfolio_value', 'N/A'):.2f}"
            log_data['profit'] = f"${sim_account.get('profit_loss', 'N/A'):.2f}"
            log_data['prices_list'] = analysis_result.get('prices') if isinstance(analysis_result.get('prices'), list) else []

            write_trade_log(log_data) # Write the consolidated log entry

        else:
            # Handle case where the submitted stock symbol isn't in our configured list
            flash(f"Invalid stock symbol '{chosen_symbol}' selected.", "warning")

    # --- Render Template (for GET requests or after POST) ---
    trade_log_entries = read_trade_log() # Get latest log entries
    # Pass all necessary data to the HTML template
    return render_template(
        "dashboard_v2.html",
        symbols=SYMBOLS,
        analysis_result=analysis_result, # Analysis results (or None on GET)
        trade_log=trade_log_entries,     # List of past trades
        live_poll_interval_ms = LIVE_PRICE_POLL_INTERVAL * 1000 # Interval for JS polling
    )

# --- API Endpoints for JavaScript ---

@app.route('/live_prices')
def get_live_prices_api():
    """API endpoint to serve the latest prices gathered by the background polling thread."""
    with price_lock: # Use lock to safely read shared data
        current_prices = latest_prices.copy()
    # log.debug(f"Serving polled prices: {current_prices}") # Usually too verbose
    return jsonify(current_prices)

@app.route('/account_details')
def get_account_details_api():
    """API endpoint to serve current simulated account details."""
    # Need latest prices to calculate portfolio value accurately
    with price_lock: # Use lock to safely read shared data
        current_prices = latest_prices.copy()
    # Get details from the simulator instance
    details = simulator.get_account_details(current_prices)
    if details:
        return jsonify(details)
    else:
        # This should generally not happen with the simulator unless there's an internal error
        log.error("Failed to get simulated account details.")
        return jsonify({"error": "Could not get simulated account details."}), 500


# --- Main Execution Guard ---
if __name__ == "__main__":
    log.info("Starting Flask Trading Robot V2 (yfinance + Simulator)...")
    # Run the Flask development server
    # debug=False is recommended when running background threads to avoid duplicate execution
    # use_reloader=False also prevents the background thread from starting twice
    # host='0.0.0.0' makes the app accessible on your local network (optional)
    app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)













