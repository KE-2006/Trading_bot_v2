# TradingRobotTeamV2/utils/portfolio_simulator.py
import logging
from typing import Dict, Optional, List, Any # <--- ADDED 'Any' HERE
import pandas as pd
from datetime import datetime

# Import config for starting cash
from config import SIM_STARTING_CASH, SYMBOLS

log = logging.getLogger(__name__)

class PortfolioSimulator:
    """
    Simulates a simple trading portfolio in memory.
    Tracks cash, positions (shares owned), and calculates portfolio value.
    Does NOT simulate fees, slippage, or complex order types.
    """
    def __init__(self, starting_cash: float = SIM_STARTING_CASH, symbols: List[str] = SYMBOLS):
        """
        Initializes the simulator.

        Args:
            starting_cash: The initial amount of fake cash.
            symbols: List of symbols the simulator should be aware of.
        """
        self.cash = starting_cash
        self.starting_cash = starting_cash
        # Dictionary to store quantity of shares owned for each symbol
        # Example: {'AAPL': 10, 'MSFT': 5}
        self.positions: Dict[str, int] = {symbol: 0 for symbol in symbols}
        # List to store records of simulated trades
        self.trade_history: List[Dict[str, Any]] = [] # Hint that dict values can be Any type
        log.info(f"PortfolioSimulator initialized with starting cash: ${starting_cash:.2f}")

    def get_current_position(self, symbol: str) -> int:
        """Returns the number of shares currently held for a symbol."""
        return self.positions.get(symbol, 0)

    def get_portfolio_value(self, latest_prices: Dict[str, float]) -> float:
        """
        Calculates the total value of the simulated portfolio (cash + holdings).

        Args:
            latest_prices: A dictionary mapping symbols to their latest known price.
                           Example: {'AAPL': 150.50, 'MSFT': 280.00}

        Returns:
            The total portfolio value. Returns only cash value if prices are missing.
        """
        holdings_value = 0.0
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                current_price = latest_prices.get(symbol)
                if current_price and current_price > 0:
                    holdings_value += quantity * current_price
                else:
                    log.warning(f"Missing latest price for {symbol} in get_portfolio_value. Holdings value may be inaccurate.")

        total_value = self.cash + holdings_value
        return total_value

    # Hint that dict values can be Any type
    def get_account_details(self, latest_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Returns a dictionary summarizing the simulated account status.
        Mimics the structure returned by the old AlpacaTrader for compatibility.
        """
        portfolio_value = self.get_portfolio_value(latest_prices)
        profit_loss = portfolio_value - self.starting_cash

        details = {
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "equity": portfolio_value, # In simple sim, equity = portfolio value
            "long_market_value": portfolio_value - self.cash, # Value of shares held
            "short_market_value": 0.0, # Not simulating shorting
            "buying_power": self.cash, # Simple sim: buying power = cash
            "daytrade_count": 0, # Not tracked
            "account_blocked": False, # Not simulated
            "trading_blocked": False, # Not simulated
            "account_status": "ACTIVE",
            "profit_loss": profit_loss, # Add profit/loss calculation
            "starting_cash": self.starting_cash
        }
        return details

    # Hint that dict values can be Any type
    def simulate_trade(self, action: str, symbol: str, qty: int, price: Optional[float]) -> Dict[str, Any]:
        """
        Simulates executing a trade (BUY or SELL).

        Args:
            action: "BUY" or "SELL".
            symbol: The stock symbol.
            qty: The number of shares.
            price: The price at which the trade is simulated. If None, trade fails.

        Returns:
            A dictionary indicating success/failure and a message.
            Mimics the structure returned by the old AlpacaTrader for compatibility.
        """
        if price is None or price <= 0:
            log.warning(f"Simulated trade failed for {symbol}: Invalid price ({price}).")
            return {"success": False, "message": f"Trade failed: Invalid price ({price}).", "order_id": None, "error": "InvalidPrice"}

        cost = qty * price
        current_shares = self.get_current_position(symbol)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if action == "BUY":
            if self.cash >= cost:
                self.cash -= cost
                self.positions[symbol] = current_shares + qty
                message = f"Simulated BUY {qty} {symbol} @ ${price:.2f}. Cost: ${cost:.2f}. Cash left: ${self.cash:.2f}."
                log.info(message)
                self.trade_history.append({
                    "timestamp": timestamp, "symbol": symbol, "action": "BUY",
                    "quantity": qty, "price": price, "cost": cost
                })
                return {"success": True, "message": message, "order_id": f"SIM-{len(self.trade_history)}", "error": None}
            else:
                message = f"Simulated BUY failed: Insufficient cash for {qty} {symbol} @ ${price:.2f}. Need ${cost:.2f}, have ${self.cash:.2f}."
                log.warning(message)
                return {"success": False, "message": message, "order_id": None, "error": "InsufficientFunds"}

        elif action == "SELL":
            if current_shares >= qty:
                self.cash += cost
                self.positions[symbol] = current_shares - qty
                message = f"Simulated SELL {qty} {symbol} @ ${price:.2f}. Proceeds: ${cost:.2f}. Cash now: ${self.cash:.2f}."
                log.info(message)
                self.trade_history.append({
                    "timestamp": timestamp, "symbol": symbol, "action": "SELL",
                    "quantity": qty, "price": price, "proceeds": cost
                })
                return {"success": True, "message": message, "order_id": f"SIM-{len(self.trade_history)}", "error": None}
            else:
                message = f"Simulated SELL failed: Cannot sell {qty} {symbol}, only own {current_shares}."
                log.warning(message)
                return {"success": False, "message": message, "order_id": None, "error": "InsufficientShares"}

        else:
            log.warning(f"Invalid action '{action}' for simulate_trade.")
            return {"success": False, "message": f"Invalid action: {action}", "order_id": None, "error": "InvalidAction"}

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = PortfolioSimulator(starting_cash=10000, symbols=['XYZ', 'ABC'])

    print("\n--- Initial State ---")
    print(sim.get_account_details(latest_prices={'XYZ': 100, 'ABC': 50}))

    print("\n--- Simulate Trades ---")
    buy_result1 = sim.simulate_trade("BUY", "XYZ", 10, 105.00) # Buy 10 XYZ @ 105
    print(f"Buy 1 Result: {buy_result1}")
    buy_result2 = sim.simulate_trade("BUY", "ABC", 5, 52.00)  # Buy 5 ABC @ 52
    print(f"Buy 2 Result: {buy_result2}")
    sell_result1 = sim.simulate_trade("SELL", "XYZ", 5, 110.00) # Sell 5 XYZ @ 110
    print(f"Sell 1 Result: {sell_result1}")
    buy_fail = sim.simulate_trade("BUY", "XYZ", 100, 100.00) # Try to buy too much
    print(f"Buy Fail Result: {buy_fail}")
    sell_fail = sim.simulate_trade("SELL", "ABC", 10, 55.00) # Try to sell too much
    print(f"Sell Fail Result: {sell_fail}")

    print("\n--- Final State ---")
    # Need current prices to value portfolio
    current_p = {'XYZ': 112.00, 'ABC': 53.50}
    print(f"Positions: {sim.positions}")
    print(sim.get_account_details(latest_prices=current_p))

    print("\n--- Trade History ---")
    for trade in sim.trade_history:
        print(trade)

