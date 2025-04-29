# TradingRobotTeamV2/app.py (yfinance + Simulator Version)
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pandas as pd
import datetime
import threading # For background polling thread
import time
import os
import logging
from typing import Dict, Optional, Any # Make sure Any and Dict are imported

# Import project components
from config import (
    SYMBOLS, TRADE_LOG_FILE, NEWSAPI_KEY, # No longer need Alpaca keys here
    PRICE_BOT_INPUT_DAYS, PRICE_BOT_SMA_PERIOD, PRICE_BOT_MIN_TRAINING_DAYS,
    LIVE_PRICE_POLL_INTERVAL, # How often to poll yfinance
    log # Use logger from config
)
from agents.price_bot_v2 import PriceBotV2
from agents.news_bot import NewsBot
# Ensure you are importing the correct TradeBot version
# Use the one with relaxed rules and potentially debug prints if needed
from agents.trade_bot_v2 import TradeBotV2
from utils.data_fetcher_v2 import ( # Use the yfinance data fetcher
    get_historical_prices_for_training,
    get_recent_prices_for_prediction,
    fetch_news_headlines,
    get_latest_quote_yf # Function to get latest quote
)
# Import the new simulator, remove AlpacaTrader
from utils.portfolio_simulator import PortfolioSimulator

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flash messages

# --- Global Variables & Initialization ---
# Initialize Portfolio Simulator (replaces AlpacaTrader)
simulator = PortfolioSimulator(symbols=SYMBOLS) # Uses starting cash from config

# Initialize Bots (using yfinance data fetcher functions now)
bots = {}
for symbol in SYMBOLS:
    bots[symbol] = {
        "price_bot": PriceBotV2(input_days=PRICE_BOT_INPUT_DAYS, sma_period=PRICE_BOT_SMA_PERIOD),
        "news_bot": NewsBot(),
        "trade_bot": TradeBotV2(safety_on=True) # Make sure this matches the file you intend to use
    }
log.info(f"Initialized V2 bots for symbols: {list(bots.keys())}")

# Live price storage (will be updated by polling thread)
latest_prices = {symbol: 0.0 for symbol in SYMBOLS}
# Lock for safely updating/reading latest_prices from multiple threads
price_lock = threading.Lock()

# --- Background Polling for "Live" Prices ---
def poll_live_prices():
    """Periodically fetches latest quotes using yfinance."""
    global latest_prices
    log.info("Starting background thread for polling yfinance quotes...")
    while True:
        log.debug(f"Polling yfinance for latest quotes for: {SYMBOLS}")
        temp_prices = {}
        for symbol in SYMBOLS:
            quote = get_latest_quote_yf(symbol)
            if quote is not None:
                temp_prices[symbol] = quote
            else:
                log.warning(f"Polling failed to get quote for {symbol}, using previous value.")
                with price_lock: # Need lock to safely read previous value
                     # Use get() with default to avoid KeyError if symbol somehow missing
                     temp_prices[symbol] = latest_prices.get(symbol, 0.0)

            time.sleep(0.5) # Small delay between symbols to avoid hammering API

        # Update the global dictionary safely
        with price_lock:
            latest_prices.update(temp_prices)
        log.debug(f"Updated latest_prices: {latest_prices}")

        # Wait for the configured interval
        # Ensure interval is positive to avoid issues
        poll_interval_seconds = max(10, LIVE_PRICE_POLL_INTERVAL) # Minimum 10 seconds
        time.sleep(poll_interval_seconds)


# Start the polling thread
polling_thread = threading.Thread(target=poll_live_prices, daemon=True)
polling_thread.start()


# --- Helper Functions ---
def read_trade_log():
    """Reads and parses the trade log file."""
    log_entries = []
    if not os.path.exists(TRADE_LOG_FILE): return log_entries
    try:
        with open(TRADE_LOG_FILE, "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                parts = line.strip().split(" | ")
                if len(parts) >= 5:
                    entry = { "timestamp": parts[0], "symbol": parts[1],
                              "decision": parts[3].split(': ')[-1], "result": parts[4].split(': ')[-1],
                              "details": " | ".join(parts[5:]) if len(parts) > 5 else "" }
                    log_entries.append(entry)
    except Exception as e: log.error(f"Error reading trade log '{TRADE_LOG_FILE}': {e}")
    return log_entries

def write_trade_log(log_data: Dict):
    """Appends a structured entry to the trade log file."""
    try:
        # Use data directly from the simulator/analysis result
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
            f"Signals: P={log_data.get('price_signal_text','?')}, N={log_data.get('news_signal_text','?')}, SMA={'A' if log_data.get('price_above_sma') else 'B' if log_data.get('price_above_sma') is False else '?'}\n"
        )
        with open(TRADE_LOG_FILE, "a") as diary: diary.write(log_line)
        log.info(f"Trade log entry written for {log_data.get('symbol', 'N/A')}.")
    except Exception as e: log.error(f"Error writing trade log: {e}")


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    """Handles the main dashboard display and trade analysis requests."""
    analysis_result = None # Store results from POST request

    if request.method == 'POST' and request.form.get('action') == 'analyze_trade':
        chosen_symbol = request.form.get('chosen_stock')
        if chosen_symbol in SYMBOLS:
            log.info(f"--- yfinance/Sim Analysis & Trade Action for {chosen_symbol} ---")
            analysis_result = {'symbol': chosen_symbol}

            # 1. Get Data (using yfinance fetcher)
            # Calculate days needed based on bot's internal requirement
            days_needed_for_pred = bots[chosen_symbol]["price_bot"].min_data_length
            log.debug(f"Fetching {days_needed_for_pred} recent daily prices for {chosen_symbol}...")
            # Pass symbol and days needed to the yfinance fetcher
            prices_series = get_recent_prices_for_prediction(chosen_symbol, days_needed=days_needed_for_pred)

            log.debug(f"Fetching news headlines for {chosen_symbol}...")
            news_headlines = fetch_news_headlines(chosen_symbol)
            analysis_result['news'] = news_headlines

            # Store prices as list for display, keep series for calcs
            # Use .tolist() safely, checking if prices_series is not None
            analysis_result['prices'] = prices_series.tolist() if prices_series is not None else "Not available"

            # Check if price data is sufficient
            if prices_series is None:
                 error_msg = f"Insufficient recent price data for {chosen_symbol} analysis (yfinance)."
                 log.warning(error_msg)
                 analysis_result['error'] = error_msg
                 analysis_result['decision'] = "HOLD" # Cannot decide without data
                 analysis_result['trade_result'] = {"success": False, "message": "Analysis failed: Insufficient price data.", "order_id": None, "error": "InsufficientData"}

            else:
                # 2. Get Bot & SMA Signals
                price_bot = bots[chosen_symbol]["price_bot"]
                news_bot = bots[chosen_symbol]["news_bot"]
                trade_bot = bots[chosen_symbol]["trade_bot"]

                # Ensure PriceBot is trained (or train it now)
                if not price_bot.is_trained:
                    log.info(f"PriceBotV2 for {chosen_symbol} not trained. Training now...")
                    # Use the correct function for training data
                    hist_prices_series = get_historical_prices_for_training(chosen_symbol)
                    if hist_prices_series is not None:
                        price_bot.train(hist_prices_series.tolist()) # Train expects list
                    else:
                         log.warning(f"Could not train PriceBotV2 for {chosen_symbol} - yfinance history fetch failed.")

                # Get signals
                # Predict expects list
                price_signal = price_bot.predict(prices_series.tolist())
                news_signal_raw = news_bot.analyze(news_headlines)
                news_signal = 0 if news_signal_raw is None or not news_headlines else news_signal_raw # Default to neutral

                # --- Calculate SMA signal with more logging ---
                price_above_sma = None # Default to None
                current_sma = None
                latest_price = None
                try:
                    # *** ADDED LOGGING HERE ***
                    log.debug(f"Calculating SMA signal for {chosen_symbol}. Data length received: {len(prices_series)}, Required SMA Period: {PRICE_BOT_SMA_PERIOD}")
                    # Check if we have AT LEAST enough data points for the SMA window
                    if len(prices_series) >= PRICE_BOT_SMA_PERIOD:
                         sma_series = prices_series.rolling(window=PRICE_BOT_SMA_PERIOD).mean()
                         current_sma = sma_series.iloc[-1] # Get the last calculated SMA value
                         latest_price = prices_series.iloc[-1]
                         # *** ADDED LOGGING HERE ***
                         log.debug(f"SMA Calculation: Latest Price = {latest_price}, Calculated SMA = {current_sma}")
                         # Check if BOTH values are valid numbers before comparing
                         if pd.notna(current_sma) and pd.notna(latest_price):
                              price_above_sma = latest_price > current_sma
                              # *** ADDED LOGGING HERE ***
                              log.debug(f"SMA Result: PriceAboveSMA = {price_above_sma}")
                         else:
                             # *** ADDED LOGGING HERE ***
                             log.warning(f"Could not compare Price vs SMA for {chosen_symbol} (SMA or Price is NaN). SMA Series Tail:\n{sma_series.tail()}")
                             price_above_sma = None # Explicitly set to None if NaN
                    else:
                         # *** ADDED LOGGING HERE ***
                         log.warning(f"Not enough data ({len(prices_series)}) for SMA({PRICE_BOT_SMA_PERIOD}) calculation for {chosen_symbol}.")
                         price_above_sma = None # Explicitly set to None if not enough data
                except Exception as e:
                    log.error(f"Error calculating SMA signal for {chosen_symbol}: {e}", exc_info=True)
                    price_above_sma = None # Ensure it's None on any error
                # --- End of SMA calculation block ---

                analysis_result['price_signal'] = price_signal
                analysis_result['news_signal'] = news_signal
                analysis_result['price_above_sma'] = price_above_sma # Will be None if calculation failed
                analysis_result['price_signal_text'] = "Up" if price_signal == 1 else "Down" if price_signal == 0 else "N/A"
                analysis_result['news_signal_text'] = "Good" if news_signal == 1 else "Bad" if news_signal == -1 else "Neutral"
                log.info(f"{chosen_symbol} Signals - Price: {analysis_result['price_signal_text']}, News: {analysis_result['news_signal_text']}, SMA: {'Above' if price_above_sma else 'Below' if price_above_sma is False else 'N/A'}")

                # 3. Get Trade Decision
                # Decide expects list
                decision = trade_bot.decide(price_signal, news_signal, price_above_sma, prices_series.tolist())
                analysis_result['decision'] = decision
                log.info(f"{chosen_symbol} Trade Decision: {decision}")

                # 4. Execute Trade using Simulator
                trade_result_dict = {"success": True, "message": "Decision is HOLD.", "order_id": None, "error": None}
                if decision in ["BUY", "SELL"]:
                    with price_lock: # Safely read latest price
                        sim_price = latest_prices.get(chosen_symbol)
                    if sim_price and sim_price > 0:
                        log.info(f"Simulating {decision} for {chosen_symbol} @ price {sim_price:.2f}...")
                        trade_result_dict = simulator.simulate_trade(decision, chosen_symbol, 1, sim_price) # Use default qty=1 for now
                    else:
                        log.warning(f"Cannot simulate {decision} for {chosen_symbol}: Latest price unknown or invalid ({sim_price}).")
                        trade_result_dict = {"success": False, "message": f"Simulation failed: Price unavailable ({sim_price}).", "order_id": None, "error": "PriceUnavailable"}
                    log.info(f"{chosen_symbol} Simulation Result: {trade_result_dict}")
                analysis_result['trade_result'] = trade_result_dict

            # 5. Log the action using simulator state
            # Get current simulator state AFTER potential trade
            with price_lock: # Need lock to get consistent portfolio value
                current_prices_for_log = latest_prices.copy()
            sim_account = simulator.get_account_details(current_prices_for_log)
            sim_position = simulator.get_current_position(chosen_symbol)

            log_data = analysis_result.copy()
            log_data['trade_result_msg'] = analysis_result['trade_result']['message'] # Log the message part
            log_data['cash'] = f"${sim_account.get('cash', 'N/A'):.2f}"
            log_data['shares'] = sim_position
            log_data['portfolio'] = f"${sim_account.get('portfolio_value', 'N/A'):.2f}"
            log_data['profit'] = f"${sim_account.get('profit_loss', 'N/A'):.2f}"
            # Pass the original prices list (if available) for logging length
            log_data['prices_list'] = analysis_result.get('prices') if isinstance(analysis_result.get('prices'), list) else []


            write_trade_log(log_data)

        else:
            flash(f"Invalid stock symbol '{chosen_symbol}' selected.", "warning")

    # --- For GET requests or after POST ---
    trade_log_entries = read_trade_log()
    # --- ENSURE VARIABLE IS PASSED ON GET REQUESTS TOO ---
    return render_template(
        "dashboard_v2.html", # Make sure using the V2 template
        symbols=SYMBOLS,
        analysis_result=analysis_result, # Pass analysis result (might be None on GET)
        trade_log=trade_log_entries,
        live_poll_interval_ms = LIVE_PRICE_POLL_INTERVAL * 1000 # Pass interval always
    )

@app.route('/live_prices')
def get_live_prices_api():
    """API endpoint for polled 'live' prices."""
    with price_lock: # Safely copy the dictionary
        current_prices = latest_prices.copy()
    # log.debug(f"Serving polled prices: {current_prices}") # Can be noisy
    return jsonify(current_prices)

@app.route('/account_details')
def get_account_details_api():
    """API endpoint for simulated account details."""
    # Need latest prices to calculate portfolio value accurately
    with price_lock:
        current_prices = latest_prices.copy()
    details = simulator.get_account_details(current_prices)
    if details:
        return jsonify(details)
    else:
        # Should generally not happen with simulator unless error
        return jsonify({"error": "Could not get simulated account details."}), 500


# --- Main Execution ---
if __name__ == "__main__":
    log.info("Starting Flask Trading Robot V2 (yfinance + Simulator)...")
    # Use debug=False and use_reloader=False when running background threads
    # Ensure the script being run is app.py (if you saved it with that name)
    app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False) # Use different port maybe











