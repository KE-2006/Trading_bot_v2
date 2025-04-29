# TradingRobotTeamV2/utils/portfolio_simulator.py
"""
In-memory portfolio simulator for tracking cash and positions without a broker API.
"""

import logging
from typing import Dict, Optional, List, Any
import pandas as pd
from datetime import datetime

# Import config for starting cash and symbols list
from config import SIM_STARTING_CASH, SYMBOLS

log = logging.getLogger(__name__)

class PortfolioSimulator:
    """
    Simulates a simple trading portfolio in memory.
    Tracks cash, positions (shares owned), and calculates portfolio value.
    Does NOT simulate commissions, fees, slippage, or complex order types.
    """
    def __init__(self, starting_cash: float = SIM_STARTING_CASH, symbols: List[str] = SYMBOLS):
        """
        Initializes the simulator.

        Args:
            starting_cash (float): The initial amount of fake cash.
            symbols (List[str]): List of symbols the simulator should track positions for.
        """
        if starting_cash < 0:
            log.warning("Starting cash is negative. Setting to 0.")
            starting_cash = 0.0
        self.cash: float = starting_cash
        self.starting_cash: float = starting_cash
        # Dictionary to store quantity of shares owned for each symbol. {SYMBOL: quantity}
        self.positions: Dict[str, int] = {symbol: 0 for symbol in symbols}
        # List to store records of simulated trades for logging/review
        self.trade_history: List[Dict[str, Any]] = []
        log.info(f"PortfolioSimulator initialized with starting cash: ${starting_cash:.2f}")

    def get_current_position(self, symbol: str) -> int:
        """
        Returns the number of shares currently held for a specific symbol.

        Args:
            symbol (str): The stock ticker symbol.

        Returns:
            int: The quantity of shares held (0 if none or symbol unknown).
        """
        return self.positions.get(symbol, 0)

    def get_portfolio_value(self, latest_prices: Dict[str, float]) -> float:
        """
        Calculates the total estimated value of the simulated portfolio (cash + value of holdings).

        Args:
            latest_prices (Dict[str, float]): A dictionary mapping symbols to their latest known price.
                                              Example: {'AAPL': 150.50, 'MSFT': 280.00}

        Returns:
            float: The total estimated portfolio value. Returns only cash value if prices are missing.
        """
        holdings_value = 0.0
        for symbol, quantity in self.positions.items():
            if quantity > 0: # Only calculate value for shares we own
                current_price = latest_prices.get(symbol)
                # Ensure price is valid before calculating
                if current_price and isinstance(current_price, (int, float)) and current_price > 0:
                    holdings_value += quantity * current_price
                else:
                    # If price is missing or invalid, we can't value this holding accurately
                    log.warning(f"Missing or invalid latest price ({current_price}) for {symbol} in get_portfolio_value. Holdings value may be inaccurate.")
                    # Option: Could potentially use an average cost basis or last known price, but adds complexity.
                    # For simplicity, we currently exclude holdings with unknown prices from the value.

        total_value = self.cash + holdings_value
        return total_value

    def get_account_details(self, latest_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Returns a dictionary summarizing the simulated account status.
        Mimics the structure returned by the old AlpacaTrader for compatibility with the app.

         Args:
            latest_prices (Dict[str, float]): Dictionary mapping symbols to their latest known price.

        Returns:
            Dict[str, Any]: A dictionary containing simulated account metrics.
        """
        portfolio_value = self.get_portfolio_value(latest_prices)
        # Calculate profit/loss based on current value vs starting cash
        profit_loss = portfolio_value - self.starting_cash
        # Calculate current value of all shares held
        holdings_value = portfolio_value - self.cash

        details = {
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "equity": portfolio_value, # In this simple simulation, equity equals portfolio value
            "long_market_value": holdings_value, # Current market value of owned shares
            "short_market_value": 0.0, # This simulator doesn't handle short positions
            "buying_power": self.cash, # Simple simulation: buying power equals available cash
            "daytrade_count": 0, # Not tracked by this simulator
            "account_blocked": False, # Not simulated
            "trading_blocked": False, # Not simulated
            "account_status": "ACTIVE", # Assume always active
            "profit_loss": profit_loss, # Include calculated profit/loss
            "starting_cash": self.starting_cash
        }
        return details

    def simulate_trade(self, action: str, symbol: str, qty: int, price: Optional[float]) -> Dict[str, Any]:
        """
        Simulates executing a market trade (BUY or SELL) at a given price.
        Updates cash and position quantity. Records the trade.

        Args:
            action (str): "BUY" or "SELL".
            symbol (str): The stock ticker symbol.
            qty (int): The number of shares to trade (must be positive).
            price (Optional[float]): The price at which the trade is simulated. If None or <= 0, trade fails.

        Returns:
            Dict[str, Any]: A dictionary indicating success/failure and a message.
                            Includes 'success' (bool), 'message' (str), 'order_id' (str or None), 'error' (str or None).
        """
        # --- Input Validation ---
        if action not in ["BUY", "SELL"]:
            log.warning(f"Invalid action '{action}' for simulate_trade.")
            return {"success": False, "message": f"Invalid action: {action}", "order_id": None, "error": "InvalidAction"}
        if qty <= 0:
             log.warning(f"Simulated trade quantity must be positive, got {qty}.")
             return {"success": False, "message": f"Trade failed: Quantity must be positive ({qty}).", "order_id": None, "error": "InvalidQuantity"}
        if price is None or not isinstance(price, (int, float)) or price <= 0:
            log.warning(f"Simulated trade failed for {symbol}: Invalid price ({price}).")
            return {"success": False, "message": f"Trade failed: Invalid price ({price}).", "order_id": None, "error": "InvalidPrice"}
        if symbol not in self.positions:
             log.warning(f"Simulated trade failed: Symbol '{symbol}' not tracked by simulator.")
             return {"success": False, "message": f"Trade failed: Symbol '{symbol}' not tracked.", "order_id": None, "error": "UntrackedSymbol"}

        # --- Trade Simulation ---
        cost = qty * price # Total value of the trade
        current_shares = self.get_current_position(symbol)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if action == "BUY":
            # Check if enough cash is available
            if self.cash >= cost:
                self.cash -= cost # Decrease cash
                self.positions[symbol] = current_shares + qty # Increase shares
                message = f"Simulated BUY {qty} {symbol} @ ${price:.2f}. Cost: ${cost:.2f}. Cash left: ${self.cash:.2f}."
                log.info(message)
                # Record trade details
                self.trade_history.append({
                    "timestamp": timestamp, "symbol": symbol, "action": "BUY",
                    "quantity": qty, "price": price, "cost": cost
                })
                # Return success message and simulated order ID
                return {"success": True, "message": message, "order_id": f"SIM-{len(self.trade_history)}", "error": None}
            else:
                # Not enough cash
                message = f"Simulated BUY failed: Insufficient cash for {qty} {symbol} @ ${price:.2f}. Need ${cost:.2f}, have ${self.cash:.2f}."
                log.warning(message)
                return {"success": False, "message": message, "order_id": None, "error": "InsufficientFunds"}

        elif action == "SELL":
            # Check if enough shares are owned
            if current_shares >= qty:
                self.cash += cost # Increase cash
                self.positions[symbol] = current_shares - qty # Decrease shares
                message = f"Simulated SELL {qty} {symbol} @ ${price:.2f}. Proceeds: ${cost:.2f}. Cash now: ${self.cash:.2f}."
                log.info(message)
                # Record trade details
                self.trade_history.append({
                    "timestamp": timestamp, "symbol": symbol, "action": "SELL",
                    "quantity": qty, "price": price, "proceeds": cost
                })
                # Return success message and simulated order ID
                return {"success": True, "message": message, "order_id": f"SIM-{len(self.trade_history)}", "error": None}
            else:
                # Not enough shares to sell
                message = f"Simulated SELL failed: Cannot sell {qty} {symbol}, only own {current_shares}."
                log.warning(message)
                return {"success": False, "message": message, "order_id": None, "error": "InsufficientShares"}

# Example Usage (for testing when running this file directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Setup logging for test run
    # Use symbols from config for consistency
    sim_test = PortfolioSimulator(starting_cash=10000, symbols=SYMBOLS)

    print("\n--- Initial State ---")
    # Need some dummy prices for initial valuation
    initial_prices = {s: 100.0 for s in SYMBOLS} # Assume all start at 100
    print(sim_test.get_account_details(latest_prices=initial_prices))

    print("\n--- Simulate Trades ---")
    # Use symbols defined in config
    sym1 = SYMBOLS[0] if SYMBOLS else 'XYZ'
    sym2 = SYMBOLS[1] if len(SYMBOLS) > 1 else 'ABC'

    buy_result1 = sim_test.simulate_trade("BUY", sym1, 10, 105.00)
    print(f"Buy 1 Result ({sym1}): {buy_result1['message']}")
    buy_result2 = sim_test.simulate_trade("BUY", sym2, 5, 52.00)
    print(f"Buy 2 Result ({sym2}): {buy_result2['message']}")
    sell_result1 = sim_test.simulate_trade("SELL", sym1, 5, 110.00)
    print(f"Sell 1 Result ({sym1}): {sell_result1['message']}")
    buy_fail = sim_test.simulate_trade("BUY", sym1, 100, 100.00) # Try to buy too much
    print(f"Buy Fail Result ({sym1}): {buy_fail['message']}")
    sell_fail = sim_test.simulate_trade("SELL", sym2, 10, 55.00) # Try to sell too much
    print(f"Sell Fail Result ({sym2}): {sell_fail['message']}")

    print("\n--- Final State ---")
    # Need current prices to value portfolio
    current_prices_test = {sym1: 112.00, sym2: 53.50}
    # Add prices for other symbols if they exist in config
    for s in SYMBOLS:
         if s not in current_prices_test: current_prices_test[s] = 100.0 # Assume others unchanged

    print(f"Positions: {sim_test.positions}")
    print(sim_test.get_account_details(latest_prices=current_prices_test))

    print("\n--- Trade History ---")
    if sim_test.trade_history:
        for trade in sim_test.trade_history:
            print(trade)
    else:
        print("No trades recorded.")

