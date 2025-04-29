# Trading Robot Team V2 (Portfolio Project)

This is version 2 of the simple web-based trading bot dashboard built with Flask. It includes improvements to bot logic and UI aesthetics.

**Core Features:**

* **Smarter Agent Bots:**
    * `PriceBotV2`: Predicts price direction using Logistic Regression with features including recent price changes AND the difference from a Simple Moving Average (SMA).
    * `NewsBot`: Analyzes news sentiment using a Hugging Face transformer model.
    * `TradeBotV2`: Makes BUY/SELL/HOLD decisions based on PriceBot, NewsBot, AND whether the price is above/below its SMA.
* **Alpaca Paper Trading API:** Fetches stock data, manages the paper account, and submits orders.
* **Live Data:** Uses WebSockets for real-time trade updates.
* **Flask Web Framework:** Powers the backend and website.
* **Improved Web Dashboard:** A cleaner, more modern interface using Tailwind CSS to trigger analysis, view results, monitor live prices/charts, see account status, and check the trade log.

**Disclaimer:** This is a portfolio project for demonstration purposes. The trading logic, while improved, is still simplified and **not suitable for real financial decisions or live trading.** Web scraping is unreliable. Use at your own risk.

## Project Structure

Here's how the project folders and files are organized:

TradingRobotTeamV2/
├── agents/                 # Folder for the robot "brains"
│   ├── __init__.py         # Makes 'agents' a Python package
│   ├── news_bot.py         # Reads news headlines
│   ├── price_bot_v2.py     # Predicts price using SMA
│   └── trade_bot_v2.py     # Makes the final decision
├── utils/                  # Folder for helper code
│   ├── __init__.py         # Makes 'utils' a Python package
│   ├── data_fetcher_v2.py  # Gets price/news data
│   └── trading_logic.py    # Handles Alpaca trading actions
├── templates/              # Folder for the website's HTML page
│   └── dashboard_v2.html   # The main dashboard page
├── .env                    # <-- YOU MUST CREATE THIS FILE for API keys (Keep it secret!)
├── .gitignore              # Tells Git which files to ignore
├── app_v2.py               # Main Flask web application code
├── config.py               # Settings for the project
├── requirements.txt        # List of tools needed (Python libraries)
└── trade_log_v2.txt        # Records the robot's actions (created automatically)
└── venv/                   # Your Python virtual environment (optional)


## Setup (Easy Steps!)

1.  **Get the Code:** Download or copy all the project files into a new folder named `TradingRobotTeamV2`.
2.  **Open Terminal:** Open your computer's command prompt or terminal. Use the `cd` command to go inside the `TradingRobotTeamV2` folder you just created.
    ```bash
    cd path/to/your/TradingRobotTeamV2
    ```
    *(Replace `path/to/your/TradingRobotTeamV2` with the actual path on your computer)*.
3.  **(Optional but Recommended) Make a Python Play Area:** This keeps the tools for this project separate. In the terminal, run:
    ```bash
    python -m venv venv
    ```
    Then, turn it on:
    * **Windows Command Prompt:** `.\venv\Scripts\activate.bat`
    * **Windows PowerShell:** `.\venv\Scripts\Activate.ps1` (If you get an error, use Command Prompt instead!)
    * **Mac/Linux:** `source venv/bin/activate`
    *(You should see `(venv)` at the start of your terminal line if it worked)*.
4.  **Install Tools:** Make sure your play area `(venv)` is active. Then run:
    ```bash
    pip install -r requirements.txt
    ```
    *(This downloads Flask, Alpaca tools, the robot brains, etc. It might take a minute!)*
5.  **Add Your Secret Keys:**
    * Inside your `TradingRobotTeamV2` folder, create a new file named exactly `.env` (starts with a dot).
    * Copy and paste this text into the `.env` file:
        ```dotenv
        # TradingRobotTeamV2/.env
        # IMPORTANT: Replace placeholders with your REAL Alpaca PAPER TRADING keys!
        ALPACA_API_KEY=YOUR_PAPER_API_KEY_HERE
        ALPACA_SECRET_KEY=YOUR_PAPER_SECRET_KEY_HERE
        ```
    * **CRITICAL:** Go to your Alpaca account (**Paper Trading** section), find your API keys, and replace `YOUR_PAPER_API_KEY_HERE` and `YOUR_PAPER_SECRET_KEY_HERE` with your actual keys.
    * Save the `.env` file. Make sure it's listed in `.gitignore` so you don't share your secrets!

## Run Your Robot Dashboard!

1.  Make sure you are in the `TradingRobotTeamV2` folder in your terminal (and `(venv)` is active if you used it).
2.  Make sure your `.env` file has your keys.
3.  Type this command and press Enter:
    ```bash
    flask run --port 5001
    ```
    *(Using port 5001 just helps avoid conflicts if you have other things running)*.
4.  Open your web browser (like Chrome, Firefox, or Edge) and go to this address: `http://127.0.0.1:5001`

## How to Use the Dashboard

1.  You'll see the new, cleaner dashboard website.
2.  In the "Control Panel" card, pick a stock symbol (like AAPL or TSLA) from the dropdown menu.
3.  Click the blue "Analyze & Decide" button.
4.  Look at the "Latest Analysis" card:
    * It shows what the PriceBot and NewsBot think.
    * It shows the final BUY, SELL, or HOLD decision.
    * It tells you if a practice trade was sent to your Alpaca paper account.
5.  Check the "Live Prices & Chart" card to see stock prices update automatically.
6.  Look at the "Account Status" card to see your paper trading account balance.
7.  See the history of the robot's actions in the "Trade Log" card.

Have fun experimenting with your improved Trading Robot Team! Remember, it's just for learning and practice.


