# TradingRobotTeamV2/agents/trade_bot_v2.py
from typing import List, Optional
import logging
import numpy as np # Import numpy to handle numpy bool type if needed

log = logging.getLogger(__name__)

class TradeBotV2:
    """
    Decides BUY/SELL/HOLD based on PriceBot prediction, NewsBot sentiment,
    and whether the current price is above/below its Simple Moving Average (SMA).
    Version with fixed boolean checks (using == instead of is).
    """
    def __init__(self, safety_on: bool = True):
        self.safety_button = "ON" if safety_on else "OFF"
        log.info(f"TradeBotV2 initialized with safety_button={self.safety_button}")

    def decide(self,
               price_signal: Optional[int],
               news_signal: Optional[int],
               price_above_sma: Optional[bool], # Can be bool or numpy.bool_
               recent_prices: List[float]
               ) -> str:

        # print("\n *** DEBUG: Inside TradeBotV2 decide() function (Fixed Boolean Check) *** \n") # Optional: Keep for one more test run
        log.debug("--- Entered TradeBotV2 decide() ---")

        # --- Input Validation ---
        # Use 'is None' for checking None type specifically
        if price_signal is None or news_signal is None or price_above_sma is None:
            log.warning("TradeBotV2 received invalid signals (None). Defaulting to HOLD.")
            # print("--- DEBUG: Returning HOLD due to invalid signals ---")
            return "HOLD"
        if not recent_prices or len(recent_prices) < 2:
             log.warning("TradeBotV2 requires at least 2 recent prices. Defaulting to HOLD.")
             # print("--- DEBUG: Returning HOLD due to insufficient prices ---")
             return "HOLD"

        # --- Decision Logic (Safety ON - Relaxed News Condition) ---
        if self.safety_button == "ON":
            latest_price = recent_prices[-1]
            previous_price = recent_prices[-2]
            price_moved_up = latest_price > previous_price

            log.info(f"TradeBotV2 Inputs: PriceSignal={price_signal}, NewsSignal={news_signal}, PriceAboveSMA={price_above_sma}, PriceMovedUp={price_moved_up}")

            # --- FIX: Use '== True' and '== False' for boolean comparison ---
            # This handles both standard Python bool and numpy.bool_ correctly

            # --- Relaxed Buy Conditions ---
            if price_signal == 1 and news_signal >= 0 and price_above_sma == True:
                log.info("TradeBotV2 Decision: BUY (Signal: PricePred Up, News Good/Neutral, Price > SMA)")
                # print("--- DEBUG: Returning BUY from Relaxed Buy Condition ---")
                return "BUY"

            # --- Relaxed Sell Conditions ---
            elif price_signal == 0 and news_signal <= 0 and price_above_sma == False:
                log.info("TradeBotV2 Decision: SELL (Signal: PricePred Down, News Bad/Neutral, Price < SMA)")
                # print("--- DEBUG: Returning SELL from Relaxed Sell Condition ---")
                return "SELL"

            # --- Hold if signals conflict or are weak ---
            elif price_signal == 1 and price_above_sma == False:
                 log.info("TradeBotV2 Decision: HOLD (Conflicting: PricePred Up, but Price < SMA)")
                 # print("--- DEBUG: Returning HOLD from Conflicting (Up, <SMA) ---")
                 return "HOLD"
            elif price_signal == 0 and price_above_sma == True:
                 log.info("TradeBotV2 Decision: HOLD (Conflicting: PricePred Down, but Price > SMA)")
                 # print("--- DEBUG: Returning HOLD from Conflicting (Down, >SMA) ---")
                 return "HOLD"
            elif news_signal == 1 and price_above_sma == False:
                 log.info("TradeBotV2 Decision: HOLD (Conflicting: News Good, but Price < SMA)")
                 # print("--- DEBUG: Returning HOLD from Conflicting (News Good, <SMA) ---")
                 return "HOLD"
            elif news_signal == -1 and price_above_sma == True:
                  log.info("TradeBotV2 Decision: HOLD (Conflicting: News Bad, but Price > SMA)")
                  # print("--- DEBUG: Returning HOLD from Conflicting (News Bad, >SMA) ---")
                  return "HOLD"

            # --- Default to HOLD ---
            else:
                log.info("TradeBotV2 Decision: HOLD (No strong BUY/SELL signal alignment or other HOLD condition met)")
                # print("--- DEBUG: Returning HOLD from Default Else Condition ---")
                return "HOLD"

        # --- Decision Logic (Safety OFF - Example: Simpler Rules) ---
        else: # safety_button == "OFF"
            log.info("TradeBotV2 Decision (Safety OFF): Using simpler rules.")
            if price_signal == 1 and news_signal >= 0:
                # print("--- DEBUG: Returning BUY from Safety OFF Condition ---")
                return "BUY"
            elif price_signal == 0 and news_signal <= 0:
                # print("--- DEBUG: Returning SELL from Safety OFF Condition ---")
                return "SELL"
            else:
                # print("--- DEBUG: Returning HOLD from Safety OFF Condition ---")
                return "HOLD"

# Example Usage (Keep as is)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot_v2 = TradeBotV2(safety_on=True)
    prices = [100, 101, 102, 101, 103]
    print("\n--- Testing TradeBotV2 (Safety ON) ---")
    # Use numpy bool for testing the fix
    above_sma_true_np = np.bool_(True)
    above_sma_false_np = np.bool_(False)

    decision1 = bot_v2.decide(price_signal=1, news_signal=1, price_above_sma=above_sma_true_np, recent_prices=prices)
    print(f"Test 1 (Strong Buy Signal): {decision1}") # Expect BUY
    decision2 = bot_v2.decide(price_signal=0, news_signal=-1, price_above_sma=above_sma_false_np, recent_prices=prices)
    print(f"Test 2 (Strong Sell Signal): {decision2}") # Expect SELL
    decision3 = bot_v2.decide(price_signal=1, news_signal=0, price_above_sma=above_sma_true_np, recent_prices=prices)
    print(f"Test 3 (Buy Signal, Neutral News): {decision3}") # Expect BUY
    decision4 = bot_v2.decide(price_signal=0, news_signal=0, price_above_sma=above_sma_false_np, recent_prices=prices)
    print(f"Test 4 (Sell Signal, Neutral News): {decision4}") # Expect SELL
    decision5 = bot_v2.decide(price_signal=1, news_signal=1, price_above_sma=above_sma_false_np, recent_prices=prices)
    print(f"Test 5 (Conflicting SMA): {decision5}") # Expect HOLD
    decision6 = bot_v2.decide(price_signal=0, news_signal=-1, price_above_sma=above_sma_true_np, recent_prices=prices)
    print(f"Test 6 (Conflicting SMA): {decision6}") # Expect HOLD
    decision7 = bot_v2.decide(price_signal=None, news_signal=0, price_above_sma=True, recent_prices=prices)
    print(f"Test 7 (Invalid Price Signal): {decision7}") # Expect HOLD





