# TradingRobotTeamV2/agents/trade_bot_v2.py
"""
TradeBot Agent (V2).
Makes final BUY/SELL/HOLD decisions based on aggregated signals.
"""

from typing import List, Optional
import logging
import numpy as np # Import numpy to handle numpy bool type

log = logging.getLogger(__name__)

class TradeBotV2:
    """
    Decides BUY/SELL/HOLD based on PriceBot prediction, NewsBot sentiment,
    and whether the current price is above/below its Simple Moving Average (SMA).
    Uses slightly relaxed rules (neutral news doesn't block trades).
    Uses '== True' / '== False' for robust boolean checking.
    """
    def __init__(self, safety_on: bool = True):
        """
        Initializes the TradeBotV2.

        Args:
            safety_on (bool): If True, uses the primary logic requiring signal alignment.
                              If False, uses simpler example logic (can be expanded).
        """
        self.safety_button = "ON" if safety_on else "OFF"
        log.info(f"TradeBotV2 initialized with safety_button={self.safety_button}")

    def decide(self,
               price_signal: Optional[int],
               news_signal: Optional[int],
               price_above_sma: Optional[bool], # Can be bool or numpy.bool_
               recent_prices: List[float] # Needed for basic price movement check
               ) -> str:
        """
        Makes the final trading decision based on input signals.

        Args:
            price_signal (Optional[int]): Prediction from PriceBot (1=Up, 0=Down, None=Error).
            news_signal (Optional[int]): Sentiment from NewsBot (1=Pos, -1=Neg, 0=Neu, None=Error).
            price_above_sma (Optional[bool]): True if latest price > SMA, False if < SMA, None if N/A.
            recent_prices (List[float]): List of recent closing prices.

        Returns:
            str: The trading decision ("BUY", "SELL", or "HOLD").
        """
        log.debug("--- Entered TradeBotV2 decide() ---")

        # --- Input Validation ---
        # Check if any required signal is missing (is None)
        if price_signal is None or news_signal is None or price_above_sma is None:
            log.warning("TradeBotV2 received invalid signals (None). Defaulting to HOLD.")
            return "HOLD"
        # Check if there are enough prices for basic checks (like price_moved_up)
        if not recent_prices or len(recent_prices) < 2:
             log.warning("TradeBotV2 requires at least 2 recent prices. Defaulting to HOLD.")
             return "HOLD"

        # --- Decision Logic (Safety ON - Relaxed News Condition) ---
        if self.safety_button == "ON":
            # Get latest price movement info (mainly for logging context)
            latest_price = recent_prices[-1]
            previous_price = recent_prices[-2]
            price_moved_up = latest_price > previous_price

            log.info(f"TradeBotV2 Inputs: PriceSignal={price_signal}, NewsSignal={news_signal}, PriceAboveSMA={price_above_sma}, PriceMovedUp={price_moved_up}")

            # --- Relaxed Buy Conditions ---
            # PriceBot predicts Up AND News is NOT Negative (Good or Neutral) AND Price is Above SMA
            # Use '== True' to handle both Python bool and numpy.bool_
            if price_signal == 1 and news_signal >= 0 and price_above_sma == True:
                log.info("TradeBotV2 Decision: BUY (Signal: PricePred Up, News Good/Neutral, Price > SMA)")
                return "BUY"

            # --- Relaxed Sell Conditions ---
            # PriceBot predicts Down AND News is NOT Positive (Bad or Neutral) AND Price is Below SMA
            # Use '== False' to handle both Python bool and numpy.bool_
            elif price_signal == 0 and news_signal <= 0 and price_above_sma == False:
                log.info("TradeBotV2 Decision: SELL (Signal: PricePred Down, News Bad/Neutral, Price < SMA)")
                return "SELL"

            # --- Hold if signals conflict or are weak ---
            # These conditions check for disagreements between signals
            elif price_signal == 1 and price_above_sma == False:
                 log.info("TradeBotV2 Decision: HOLD (Conflicting: PricePred Up, but Price < SMA)")
                 return "HOLD"
            elif price_signal == 0 and price_above_sma == True:
                 log.info("TradeBotV2 Decision: HOLD (Conflicting: PricePred Down, but Price > SMA)")
                 return "HOLD"
            elif news_signal == 1 and price_above_sma == False:
                 log.info("TradeBotV2 Decision: HOLD (Conflicting: News Good, but Price < SMA)")
                 return "HOLD"
            elif news_signal == -1 and price_above_sma == True:
                  log.info("TradeBotV2 Decision: HOLD (Conflicting: News Bad, but Price > SMA)")
                  return "HOLD"

            # --- Default to HOLD ---
            # If none of the specific BUY, SELL, or conflicting HOLD conditions above were met
            else:
                log.info("TradeBotV2 Decision: HOLD (No strong BUY/SELL signal alignment or other HOLD condition met)")
                return "HOLD"

        # --- Decision Logic (Safety OFF - Example: Simpler Rules) ---
        # This part only runs if safety_button was set to "OFF" when TradeBotV2 was created
        else: # safety_button == "OFF"
            log.info("TradeBotV2 Decision (Safety OFF): Using simpler rules.")
            # Buy if PriceBot predicts Up and News isn't negative
            if price_signal == 1 and news_signal >= 0:
                return "BUY"
            # Sell if PriceBot predicts Down and News isn't positive
            elif price_signal == 0 and news_signal <= 0:
                return "SELL"
            # Hold otherwise
            else:
                return "HOLD"

# Example Usage (for testing when running this file directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Setup logging for direct run test
    bot_test = TradeBotV2(safety_on=True) # Test with safety ON
    prices_test = [100, 101, 102, 101, 103] # Example price list

    print("\n--- Testing TradeBotV2 (Safety ON) ---")
    # Use standard Python bools for testing clarity
    above_sma_true = True
    above_sma_false = False

    decision1 = bot_test.decide(price_signal=1, news_signal=1, price_above_sma=above_sma_true, recent_prices=prices_test)
    print(f"Test 1 (Inputs: P=1, N=1, SMA=T): Expected BUY -> Got {decision1}")

    decision2 = bot_test.decide(price_signal=0, news_signal=-1, price_above_sma=above_sma_false, recent_prices=prices_test)
    print(f"Test 2 (Inputs: P=0, N=-1, SMA=F): Expected SELL -> Got {decision2}")

    decision3 = bot_test.decide(price_signal=1, news_signal=0, price_above_sma=above_sma_true, recent_prices=prices_test)
    print(f"Test 3 (Inputs: P=1, N=0, SMA=T): Expected BUY -> Got {decision3}")

    decision4 = bot_test.decide(price_signal=0, news_signal=0, price_above_sma=above_sma_false, recent_prices=prices_test)
    print(f"Test 4 (Inputs: P=0, N=0, SMA=F): Expected SELL -> Got {decision4}")

    decision5 = bot_test.decide(price_signal=1, news_signal=1, price_above_sma=above_sma_false, recent_prices=prices_test)
    print(f"Test 5 (Inputs: P=1, N=1, SMA=F): Expected HOLD -> Got {decision5}")

    decision6 = bot_test.decide(price_signal=0, news_signal=-1, price_above_sma=above_sma_true, recent_prices=prices_test)
    print(f"Test 6 (Inputs: P=0, N=-1, SMA=T): Expected HOLD -> Got {decision6}")

    decision7 = bot_test.decide(price_signal=None, news_signal=0, price_above_sma=True, recent_prices=prices_test)
    print(f"Test 7 (Inputs: P=None, N=0, SMA=T): Expected HOLD -> Got {decision7}")






