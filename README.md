# Trading Robot Team V2 (Portfolio Project - yfinance Version)

This project is a simple web-based trading bot dashboard built with Flask. It demonstrates the integration of:

* **Smarter Agent Bots:**
    * `PriceBotV2`: Predicts stock price direction using Logistic Regression with features including recent price changes AND the difference from a Simple Moving Average (SMA).
    * `NewsBot`: Analyzes sentiment of news headlines using a Hugging Face transformer model (fetched via NewsAPI).
    * `TradeBotV2`: Makes a BUY/SELL/HOLD decision based on signals from the other bots and the SMA trend.
* **yfinance Library:** Fetches historical and recent stock price data from Yahoo Finance.
* **NewsAPI:** Fetches relevant news headlines for sentiment analysis.
* **Internal Portfolio Simulator:** Tracks fake cash and fake share positions in memory, simulating trades without needing a broker API.
* **Flask Web Framework:** Provides a web interface (dashboard) to interact with the system.
* **Improved Web Dashboard:** A cleaner, more modern interface using Tailwind CSS to trigger analysis, view bot decisions, see simulated trade results, monitor "live-ish" prices on a chart, view simulated portfolio status, and check the trade log.

**Disclaimer:** This is a portfolio project for demonstration purposes only. The trading logic is simplified and **not intended for real financial decisions.** Data from yfinance may be delayed. Use at your own risk.

## Project Structure

(You can copy the structure below to help create your folders)

```text
TradingRobotTeamV2/
├── agents/
│   ├── __init__.py
│   ├── news_bot.py
│   ├── price_bot_v2.py
│   └── trade_bot_v2.py
├── utils/
│   ├── __init__.py
│   ├── data_fetcher_v2.py
│   └── portfolio_simulator.py
├── templates/
│   └── dashboard_v2.html
├── .env
├── .gitignore
├── app.py
├── config.py
├── requirements.txt
└── trade_log_yf_sim.txt
└── venv/
Setup (Easy Steps!)Get the Code: Make sure you have all the correct project files (like app.py, config.py, files in agents/, utils/, templates/) inside a main folder named TradingRobotTeamV2.Open Terminal: Open your command prompt or terminal and use the cd command to go into the TradingRobotTeamV2 folder.cd path/to/your/TradingRobotTeamV2
(Optional but Recommended) Make a Python Play Area (venv):python -m venv venv
Activate it:Windows Command Prompt: .\venv\Scripts\activate.batWindows PowerShell: .\venv\Scripts\Activate.ps1 (Use Command Prompt if this gives errors)Mac/Linux: source venv/bin/activate(Look for (venv) at the start of your terminal line).Install Tools: Make sure (venv) is active. Then run:pip install -r requirements.txt
(This installs Flask, yfinance, requests, transformers, etc.)Add Your Secret NewsAPI Key:Inside your TradingRobotTeamV2 folder, create a file named exactly .env (starts with a dot).Copy and paste this text into the .env file:# TradingRobotTeamV2/.env
# Get a free key from [https://newsapi.org/](https://newsapi.org/)
NEWSAPI_KEY=YOUR_NEWSAPI_KEY_HERE

# Alpaca keys are optional now, but you can leave them if you have them
# ALPACA_API_KEY=YOUR_PAPER_API_KEY_HERE
# ALPACA_SECRET_KEY=YOUR_PAPER_SECRET_KEY_HERE
Go to https://newsapi.org/, sign up for a free key, and replace YOUR_NEWSAPI_KEY_HERE with your actual key.Save the .env file. Make sure it's listed in .gitignore!Run Your Robot Dashboard!Make sure you are in the TradingRobotTeamV2 folder in your terminal (and (venv) is active if you used it).Make sure your .env file has your NewsAPI key.Type this command and press Enter:python -m flask run --port 5001
(Or just python app.py if flask run gives issues)Open your web browser (like Chrome, Firefox, or Edge) and go to this address: http://127.0.0.1:5001How to Use the DashboardYou'll see the dashboard website.In the "Control Panel" card, pick a stock symbol.Click the blue "Analyze & Decide" button.Look at the "Latest Analysis" card:It shows the signals from PriceBot, NewsBot, and the Price vs SMA check.It shows the final BUY, SELL, or HOLD decision.It tells you if a simulated trade was recorded (e.g., "Simulated BUY...").Check the "Live-ish Prices & Chart" card to see prices update periodically (based on the polling interval in config.py).Look at the "Simulated Portfolio

