# TradingRobotTeamV2/utils/trading_logic.py
# Using the same trading logic as before.
# Handles connection, order submission, account/position retrieval.

from alpaca_trade_api.rest import REST, APIError
from typing import Optional, Dict, Any
import logging

# Import config variables
from config import DEFAULT_TRADE_QTY

log = logging.getLogger(__name__)

class AlpacaTrader:
    """
    Handles connection to Alpaca, executes paper trades, and retrieves account/position info.
    """
    def __init__(self, api_client: REST):
        """
        Initializes the trader with an authenticated Alpaca API client.

        Args:
            api_client: An initialized and authenticated alpaca_trade_api.rest.REST instance.
        """
        if not isinstance(api_client, REST):
             raise TypeError("api_client must be an authenticated Alpaca REST instance.")
        self.api = api_client
        # Store initial value for simple P/L calculation
        self.starting_portfolio_value = self._get_initial_portfolio_value()
        log.info(f"AlpacaTrader initialized. Starting Portfolio Value: ${self.starting_portfolio_value or 'N/A'}")

    def _get_initial_portfolio_value(self) -> Optional[float]:
        """Fetches the portfolio value on initialization."""
        details = self.get_account_details()
        return details.get("portfolio_value") if details else None

    def get_account_details(self) -> Optional[Dict[str, Any]]:
         """Gets current cash, portfolio value, equity, etc. from Alpaca."""
         try:
            account = self.api.get_account()
            # Convert relevant fields to float/int/bool for easier use
            details = {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "buying_power": float(account.buying_power),
                "daytrade_count": int(account.daytrade_count),
                "account_blocked": bool(account.account_blocked),
                "trading_blocked": bool(account.trading_blocked),
                "account_status": account.status # e.g., ACTIVE
            }
            log.debug(f"Fetched account details: {details}")
            return details
         except APIError as e:
            log.error(f"Alpaca API error getting account details: {e}", exc_info=True)
            return None
         except Exception as e:
            log.error(f"Unexpected error getting account details: {e}", exc_info=True)
            return None

    def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gets details about the current position held for a specific symbol."""
        try:
            position = self.api.get_position(symbol)
            details = {
                "symbol": position.symbol,
                "quantity": int(position.qty),
                "market_value": float(position.market_value),
                "average_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "unrealized_pl": float(position.unrealized_pl),
                # Convert percentage string like '0.015' to 1.5
                "unrealized_pl_percent": float(position.unrealized_plpc) * 100,
                "side": position.side # 'long' or 'short'
            }
            log.debug(f"Fetched position details for {symbol}: {details}")
            return details
        except APIError as e:
            # Position not found error code
            if e.code == 40410000:
                log.debug(f"No position found for symbol {symbol}.")
                # Return a dict indicating no position, consistent structure
                return { "symbol": symbol, "quantity": 0, "market_value": 0.0, "average_entry_price": 0.0, "current_price": 0.0, "unrealized_pl": 0.0, "unrealized_pl_percent": 0.0, "side": "none" }
            else:
                log.error(f"Alpaca API error getting position for {symbol}: {e}", exc_info=True)
                return None # Indicate other error
        except Exception as e:
            log.error(f"Unexpected error getting position for {symbol}: {e}", exc_info=True)
            return None

    def submit_trade(self, action: str, symbol: str, qty: int = DEFAULT_TRADE_QTY) -> Dict[str, Any]:
        """
        Submits a market order (BUY or SELL) to Alpaca.

        Args:
            action: "BUY" or "SELL".
            symbol: The stock symbol.
            qty: The number of shares to trade.

        Returns:
            A dictionary with 'success', 'message', 'order_id', 'error'.
        """
        if action not in ["BUY", "SELL"]:
            log.warning(f"Invalid trade action '{action}' requested for {symbol}.")
            return {"success": False, "message": f"Invalid action: {action}", "order_id": None, "error": "InvalidAction"}

        log.info(f"Attempting to submit {action} order for {qty} shares of {symbol}.")

        # Pre-submission checks
        account_details = self.get_account_details()
        position_details = self.get_position_details(symbol)

        if not account_details:
             return {"success": False, "message": "Failed to get account details before trade.", "order_id": None, "error": "AccountFetchError"}
        if position_details is None: # Check for None error vs 0 quantity
             return {"success": False, "message": f"Failed to get position details for {symbol} before trade.", "order_id": None, "error": "PositionFetchError"}

        if account_details.get("trading_blocked") or account_details.get("account_blocked"):
            log.warning(f"Trade blocked for {symbol}. Account Status: {account_details.get('account_status')}, Trading Blocked: {account_details.get('trading_blocked')}")
            return {"success": False, "message": "Trade failed: Account or trading is blocked.", "order_id": None, "error": "AccountBlocked"}

        # Check sufficient shares for SELL
        if action == "SELL":
            shares_owned = position_details.get("quantity", 0)
            if shares_owned < qty:
                log.warning(f"Cannot SELL {qty} shares of {symbol}, only own {shares_owned}.")
                if shares_owned <= 0:
                     return {"success": False, "message": f"Trade failed: No shares of {symbol} owned to sell.", "order_id": None, "error": "InsufficientShares"}
                else:
                     log.info(f"Adjusting SELL quantity for {symbol} from {qty} to {shares_owned} (available).")
                     qty = shares_owned # Sell available shares

        # Submit the order (Alpaca handles buying power checks server-side)
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=action.lower(), # 'buy' or 'sell'
                type='market',
                time_in_force='gtc' # Good 'til canceled
            )
            log.info(f"Successfully submitted {action} order for {qty} {symbol}. Order ID: {order.id}")
            # Return details including the order object if needed later
            return { "success": True, "message": f"{action} order ({qty} {symbol}) submitted.", "order_id": order.id, "error": None, "order_details": order }
        except APIError as e:
            log.error(f"Alpaca API error submitting {action} order for {symbol}: {e}", exc_info=True)
            # Provide more specific error message if possible
            error_message = str(e)
            if "insufficient buying power" in error_message.lower():
                 error_code = "InsufficientFunds"
            elif "halted" in error_message.lower():
                 error_code = "TradingHalted"
            else:
                 error_code = "APIError"
            return { "success": False, "message": f"Trade failed: Alpaca API error - {e}", "order_id": None, "error": error_code }
        except Exception as e:
            log.error(f"Unexpected error submitting {action} order for {symbol}: {e}", exc_info=True)
            return { "success": False, "message": f"Trade failed: Unexpected error - {e}", "order_id": None, "error": "UnexpectedError" }

    def calculate_profit_loss(self) -> Optional[float]:
        """Calculates simple profit/loss based on current vs starting portfolio value."""
        current_details = self.get_account_details()
        # Ensure both values are valid floats before calculating
        if current_details and isinstance(self.starting_portfolio_value, float):
            current_value = current_details.get("portfolio_value")
            if isinstance(current_value, float):
                return current_value - self.starting_portfolio_value
        log.warning("Could not calculate profit/loss (missing current or starting portfolio value).")
        return None # Return None if values aren't available or valid

# Example Usage (for testing) - Requires Alpaca keys configured
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_URL
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Cannot run test: Alpaca API keys not configured.")
    else:
        api_test = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_PAPER_URL)
        trader_test = AlpacaTrader(api_test)
        test_symbol = "AAPL"

        print("\n--- Testing Account Details ---")
        acc_info = trader_test.get_account_details()
        if acc_info: print(f"Cash: ${acc_info.get('cash', 'N/A'):.2f}, Portfolio: ${acc_info.get('portfolio_value', 'N/A'):.2f}")
        else: print("Failed.")

        print(f"\n--- Testing Position Details ({test_symbol}) ---")
        pos_info = trader_test.get_position_details(test_symbol)
        if pos_info: print(f"Qty: {pos_info.get('quantity', 'N/A')}, Avg Entry: ${pos_info.get('average_entry_price', 'N/A'):.2f}")
        else: print(f"Failed or no position.")

        # --- DANGER: Uncomment to place paper trades ---
        # print(f"\n--- Testing BUY Trade ({test_symbol}) ---")
        # buy_result = trader_test.submit_trade("BUY", test_symbol, qty=1)
        # print(f"BUY Result: {buy_result}")
        # import time
        # time.sleep(5) # Wait for order
        # print(f"\n--- Testing SELL Trade ({test_symbol}) ---")
        # sell_result = trader_test.submit_trade("SELL", test_symbol, qty=1)
        # print(f"SELL Result: {sell_result}")
        # --- END DANGER ---

        print("\n--- Testing Profit/Loss Calc ---")
        pnl = trader_test.calculate_profit_loss()
        if pnl is not None: print(f"Approx P/L: ${pnl:.2f}")
        else: print("Could not calculate P/L.")
