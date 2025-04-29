# TradingRobotTeamV2/agents/price_bot_v2.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Tuple
import logging

# Get the logger we set up in config.py
log = logging.getLogger(__name__)

class PriceBotV2:
    """
    Predicts stock price direction using Logistic Regression.
    Features: Recent price changes and difference from Simple Moving Average (SMA).
    """
    def __init__(self, input_days: int = 10, sma_period: int = 20):
        """
        Initializes the improved PriceBot.

        Args:
            input_days: How many recent days of price changes to look at.
            sma_period: The period (in days) for calculating the SMA.
        """
        if input_days <= 0 or sma_period <= 0:
             raise ValueError("input_days and sma_period must be positive.")
        self.input_days = input_days
        self.sma_period = sma_period
        # Need enough data to calculate SMA and then have 'input_days' left over
        self.min_data_length = sma_period + input_days
        # Same simple brain, but we'll feed it smarter info
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='liblinear')
        self.is_trained = False
        log.info(f"PriceBotV2 initialized: InputDays={input_days}, SmaPeriod={sma_period}, MinData={self.min_data_length}")

    def _calculate_features(self, prices: pd.Series) -> Optional[pd.DataFrame]:
        """
        Calculates features for the model from a pandas Series of prices.

        Features:
        1. Percentage change for each of the last 'input_days'.
        2. Difference between the latest price and the SMA, normalized by price.

        Args:
            prices: A pandas Series of closing prices, sorted oldest to newest.

        Returns:
            A pandas DataFrame with features, or None if not enough data.
        """
        if len(prices) < self.min_data_length:
            log.warning(f"Not enough data ({len(prices)}) to calculate features. Need {self.min_data_length}.")
            return None

        try:
            features = pd.DataFrame(index=prices.index)

            # 1. Calculate recent percentage changes
            # We need 'input_days+1' prices to get 'input_days' changes
            for i in range(1, self.input_days + 1):
                # Shift(i) looks back 'i' days. pct_change calculates (price[t]/price[t-1])-1
                features[f'change_{i}d'] = prices.pct_change(periods=i).shift(-i) # Shift result back to align

            # 2. Calculate Simple Moving Average (SMA)
            features['sma'] = prices.rolling(window=self.sma_period).mean()

            # 3. Calculate difference from SMA, normalized by the price itself
            # (price - sma) / price = 1 - (sma / price)
            # Avoid division by zero or very small prices
            safe_price = prices.replace(0, np.nan) # Replace 0 with NaN temporarily
            features['sma_diff_norm'] = (safe_price - features['sma']) / safe_price
            features['sma_diff_norm'].fillna(0, inplace=True) # Fill NaN results (e.g., from division by zero) with 0

            # Drop rows with NaN values created by rolling calculations or shifts
            # We need features for the *end* of the period to predict the *next* step
            features.dropna(inplace=True)

            # Select only the feature columns needed for the model
            feature_cols = [f'change_{i}d' for i in range(1, self.input_days + 1)] + ['sma_diff_norm']
            final_features = features[feature_cols]

            log.debug(f"Calculated features shape: {final_features.shape}")
            return final_features

        except Exception as e:
            log.error(f"Error calculating features: {e}", exc_info=True)
            return None

    def train(self, historical_prices: List[float]) -> bool:
        """
        Trains the model using historical prices.

        Args:
            historical_prices: List of closing prices (oldest first).

        Returns:
            True if training succeeded, False otherwise.
        """
        self.is_trained = False # Reset trained status
        log.info(f"Attempting PriceBotV2 training with {len(historical_prices)} prices.")

        if len(historical_prices) < self.min_data_length + 1: # Need one extra point for the target 'y'
            log.warning(f"Not enough historical data ({len(historical_prices)}) for training. Need {self.min_data_length + 1}.")
            return False

        prices_series = pd.Series(historical_prices)
        features_df = self._calculate_features(prices_series)

        if features_df is None or features_df.empty:
            log.error("Feature calculation failed or yielded no data during training.")
            return False

        try:
            # Create target variable 'y': 1 if price went up *next* day, 0 otherwise
            # Align 'y' with the *end* of the feature calculation window
            # The features at index 't' predict the change from 't' to 't+1'
            price_change_next_day = prices_series.diff().shift(-1) # Change from today to tomorrow
            y = (price_change_next_day > 0).astype(int) # 1 if price went up, 0 otherwise

            # Align features (X) and target (y) by index
            aligned_data = pd.concat([features_df, y.rename('target')], axis=1).dropna()

            if aligned_data.empty:
                 log.warning("No aligned data points after combining features and target.")
                 return False

            X_train = aligned_data.drop('target', axis=1)
            y_train = aligned_data['target']

            if X_train.empty or y_train.empty:
                 log.warning("Training data (X or y) is empty after alignment.")
                 return False

            log.info(f"Training PriceBotV2 model with {X_train.shape[0]} samples.")
            self.model.fit(X_train, y_train)
            self.is_trained = True
            log.info("PriceBotV2 training complete.")
            return True

        except Exception as e:
            log.error(f"Error during PriceBotV2 model fitting: {e}", exc_info=True)
            return False

    def predict(self, recent_prices: List[float]) -> Optional[int]:
        """
        Predicts the next price direction using the latest available prices.

        Args:
            recent_prices: List of the most recent closing prices (at least min_data_length).

        Returns:
            1 (Up), 0 (Down), or None (Error/Not Trained/Insufficient Data).
        """
        if not self.is_trained:
            log.warning("PriceBotV2 predict called, but model not trained.")
            return None
        if len(recent_prices) < self.min_data_length:
            log.warning(f"PriceBotV2 predict needs {self.min_data_length} prices, got {len(recent_prices)}.")
            return None

        prices_series = pd.Series(recent_prices)
        features_df = self._calculate_features(prices_series)

        if features_df is None or features_df.empty:
            log.error("Feature calculation failed or yielded no data during prediction.")
            return None

        try:
            # Use the features from the *very last* available time point
            latest_features = features_df.iloc[[-1]] # Select last row as DataFrame

            prediction = self.model.predict(latest_features)[0]
            log.info(f"PriceBotV2 prediction: {'Up' if prediction == 1 else 'Down'}")
            return int(prediction)

        except Exception as e:
            log.error(f"Error during PriceBotV2 prediction: {e}", exc_info=True)
            return None

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Generate some fake price data for testing
    np.random.seed(42)
    base_price = 100
    price_moves = np.random.randn(100) * 0.5 # Smaller random moves
    trend = np.linspace(0, 5, 100) # Gentle upward trend
    hist_prices_test = base_price + np.cumsum(price_moves) + trend
    hist_prices_test = np.maximum(1, hist_prices_test) # Ensure price > 0

    input_d = 5
    sma_p = 10
    min_len = sma_p + input_d

    bot_v2 = PriceBotV2(input_days=input_d, sma_period=sma_p)
    trained_ok = bot_v2.train(list(hist_prices_test))

    if trained_ok:
        # Get the most recent data needed for prediction
        recent_data_for_pred = list(hist_prices_test[-min_len:])
        prediction = bot_v2.predict(recent_data_for_pred)
        if prediction is not None:
            print(f"\nPrediction based on last {min_len} prices: {'Up' if prediction == 1 else 'Down'}")
            # You could also print the features used:
            # features_for_pred = bot_v2._calculate_features(pd.Series(recent_data_for_pred))
            # if features_for_pred is not None:
            #     print("Features used for prediction:")
            #     print(features_for_pred.iloc[[-1]])
        else:
            print("\nPrediction failed.")
    else:
        print("\nTraining failed.")

