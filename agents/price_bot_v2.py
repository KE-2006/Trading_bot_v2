# TradingRobotTeamV2/agents/price_bot_v2.py
"""
PriceBot Agent using Logistic Regression.
Predicts stock price direction based on recent price changes and SMA difference.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Tuple
import logging

# Get the logger configured in config.py
log = logging.getLogger(__name__)

class PriceBotV2:
    """
    Predicts stock price direction (up/down) using a simple Logistic Regression model.
    Features: Recent price percentage changes and normalized difference from SMA.
    """
    def __init__(self, input_days: int = 10, sma_period: int = 20):
        """
        Initializes the PriceBotV2.

        Args:
            input_days (int): How many recent daily percentage changes to use as features.
            sma_period (int): The period (in days) for calculating the Simple Moving Average.
        """
        if input_days <= 0 or sma_period <= 0:
             raise ValueError("input_days and sma_period must be positive.")
        self.input_days = input_days
        self.sma_period = sma_period
        # Minimum data length needed to calculate all features for one prediction point
        self.min_data_length = sma_period + input_days
        # Logistic Regression model setup
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='liblinear')
        self.is_trained = False # Flag to track if the model has been trained
        log.info(f"PriceBotV2 initialized: InputDays={input_days}, SmaPeriod={sma_period}, MinDataRequired={self.min_data_length}")

    def _calculate_features(self, prices: pd.Series) -> Optional[pd.DataFrame]:
        """
        Calculates features for the model from a pandas Series of prices.

        Features:
        1. Percentage change for each of the last 'input_days'.
        2. Difference between the latest price and the SMA, normalized by price.

        Args:
            prices (pd.Series): Closing prices, sorted oldest to newest, with datetime index.

        Returns:
            Optional[pd.DataFrame]: DataFrame with features, or None if not enough data.
        """
        if len(prices) < self.min_data_length:
            log.warning(f"Not enough data ({len(prices)}) to calculate features. Need {self.min_data_length}.")
            return None

        try:
            features = pd.DataFrame(index=prices.index)

            # Feature 1: Recent percentage changes
            for i in range(1, self.input_days + 1):
                # Calculate % change over 'i' days and shift result to align with the *end* day
                features[f'change_{i}d'] = prices.pct_change(periods=i).shift(-i)

            # Feature 2: Simple Moving Average (SMA)
            features['sma'] = prices.rolling(window=self.sma_period).mean()

            # Feature 3: Normalized difference from SMA: (price - sma) / price
            safe_price = prices.replace(0, np.nan) # Avoid division by zero
            features['sma_diff_norm'] = (safe_price - features['sma']) / safe_price
            # Replace any NaNs created during calculation (e.g., division by zero, initial SMA values)
            # FIX for FutureWarning: Assign result back instead of using inplace=True
            features['sma_diff_norm'] = features['sma_diff_norm'].fillna(0)

            # Drop rows with any NaN values (usually at the beginning due to rolling calculations/shifts)
            features.dropna(inplace=True)

            # Select only the feature columns needed for the model prediction
            feature_cols = [f'change_{i}d' for i in range(1, self.input_days + 1)] + ['sma_diff_norm']
            final_features = features[feature_cols]

            log.debug(f"Calculated features shape: {final_features.shape}")
            return final_features

        except Exception as e:
            log.error(f"Error calculating features: {e}", exc_info=True)
            return None

    def train(self, historical_prices: List[float]) -> bool:
        """
        Trains the Logistic Regression model using historical price data.

        Args:
            historical_prices (List[float]): List of closing prices (oldest first).

        Returns:
            bool: True if training succeeded, False otherwise.
        """
        self.is_trained = False # Reset status before attempting training
        log.info(f"Attempting PriceBotV2 training with {len(historical_prices)} prices.")

        # Need enough data for feature calculation plus one extra point for the target variable
        if len(historical_prices) < self.min_data_length + 1:
            log.warning(f"Not enough historical data ({len(historical_prices)}) for training. Need {self.min_data_length + 1}.")
            return False

        # Convert list to pandas Series for easier calculations
        # Using a simple range index as dates are not strictly needed for this model's features
        prices_series = pd.Series(historical_prices)
        features_df = self._calculate_features(prices_series)

        if features_df is None or features_df.empty:
            log.error("Feature calculation failed or yielded no data during training.")
            return False

        try:
            # Create target variable 'y': 1 if price went up the *next* day, 0 otherwise
            # The features at index 't' predict the change from 't' to 't+1'
            price_change_next_day = prices_series.diff().shift(-1) # Calculate change to next day
            y = (price_change_next_day > 0).astype(int) # Convert boolean to 1 (True) or 0 (False)

            # Align features (X) and target (y) using their index. Drop rows where either is NaN.
            aligned_data = pd.concat([features_df, y.rename('target')], axis=1).dropna()

            if aligned_data.empty:
                 log.warning("No aligned data points after combining features and target for training.")
                 return False

            X_train = aligned_data.drop('target', axis=1)
            y_train = aligned_data['target']

            if X_train.empty or y_train.empty:
                 log.warning("Training data (X or y) is empty after alignment.")
                 return False

            log.info(f"Training PriceBotV2 model with {X_train.shape[0]} samples.")
            self.model.fit(X_train, y_train) # Train the model
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
            recent_prices (List[float]): List of the most recent closing prices
                                         (must contain at least min_data_length points).

        Returns:
            Optional[int]: 1 (Up), 0 (Down), or None (Error/Not Trained/Insufficient Data).
        """
        if not self.is_trained:
            log.warning("PriceBotV2 predict called, but model not trained.")
            return None
        # Need enough data to calculate features for the *last* time step
        if len(recent_prices) < self.min_data_length:
            log.warning(f"PriceBotV2 predict needs {self.min_data_length} prices, got {len(recent_prices)}.")
            return None

        prices_series = pd.Series(recent_prices)
        features_df = self._calculate_features(prices_series)

        if features_df is None or features_df.empty:
            log.error("Feature calculation failed or yielded no data during prediction.")
            return None

        try:
            # Use the features calculated for the *very last* available time point in the input data
            latest_features = features_df.iloc[[-1]] # Select last row as a DataFrame

            prediction = self.model.predict(latest_features)[0] # Predict using the last row
            prediction_int = int(prediction) # Ensure integer output
            log.info(f"PriceBotV2 prediction: {'Up' if prediction_int == 1 else 'Down'}")
            return prediction_int

        except Exception as e:
            log.error(f"Error during PriceBotV2 prediction: {e}", exc_info=True)
            return None

# Example Usage (for testing when running this file directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Setup logging for test run
    # Generate some fake price data
    np.random.seed(42)
    hist_prices_test = list(100 + np.cumsum(np.random.randn(100) * 0.5 + 0.05)) # Slight upward bias

    input_d = 5
    sma_p = 10
    min_len_test = sma_p + input_d

    bot_test = PriceBotV2(input_days=input_d, sma_period=sma_p)
    trained_ok = bot_test.train(hist_prices_test)

    if trained_ok:
        # Get the most recent data needed for prediction
        recent_data_for_pred_test = hist_prices_test[-min_len_test:]
        prediction_test = bot_test.predict(recent_data_for_pred_test)
        if prediction_test is not None:
            print(f"\nPrediction based on last {min_len_test} prices: {'Up' if prediction_test == 1 else 'Down'}")
        else:
            print("\nPrediction failed.")
    else:
        print("\nTraining failed.")

