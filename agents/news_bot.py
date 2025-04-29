# TradingRobotTeamV2/agents/news_bot.py
"""
NewsBot Agent using Hugging Face Transformers.
Analyzes sentiment of news headlines.
"""

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Optional
import logging
import torch # Explicitly import torch to check GPU availability

log = logging.getLogger(__name__)

class NewsBot:
    """
    Analyzes the sentiment of news headlines using a pre-trained transformer model.
    Provides a sentiment score: 1 (Positive), -1 (Negative), 0 (Neutral/Mixed).
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the NewsBot by loading the sentiment analysis model.

        Args:
            model_name (str): The name of the Hugging Face model/pipeline to use.
        """
        self.model_name = model_name
        self.sentiment_analyzer = None
        self.device = self._get_device() # Determine if GPU is available (-1 for CPU, 0+ for GPU)
        self._load_model()

    def _get_device(self) -> int:
        """Checks if CUDA (GPU) is available, otherwise returns device ID for CPU."""
        if torch.cuda.is_available():
            log.info("CUDA (GPU) available, NewsBot will use GPU.")
            return 0 # Default CUDA device ID
        else:
            log.info("CUDA not available, NewsBot will use CPU.")
            return -1 # Pipeline interprets -1 as CPU

    def _load_model(self):
        """Loads the Hugging Face sentiment analysis pipeline."""
        try:
            log.info(f"Loading sentiment analysis model: {self.model_name}...")
            # Load tokenizer and model explicitly for potentially better error handling
            # and resource management if needed later.
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            # Create the pipeline, specifying the task, model, tokenizer, and device
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=self.device # Use GPU if available, else CPU
            )
            log.info(f"Sentiment analysis model loaded successfully onto {'GPU' if self.device >= 0 else 'CPU'}.")
        except Exception as e:
            log.error(f"Failed to load sentiment analysis model '{self.model_name}': {e}", exc_info=True)
            # Set analyzer to None so analyze() knows the model isn't ready
            self.sentiment_analyzer = None

    def analyze(self, news_headlines: List[str]) -> Optional[int]:
        """
        Analyzes a list of news headlines and returns an overall sentiment score.

        Args:
            news_headlines (List[str]): A list of news headline strings.

        Returns:
            Optional[int]:
                1 for overall positive sentiment.
               -1 for overall negative sentiment.
                0 for neutral/mixed sentiment or if analysis fails.
               Returns None only if the model failed to load initially.
        """
        if not self.sentiment_analyzer:
            log.error("Sentiment analyzer model not loaded. Cannot analyze news.")
            return None # Indicate model loading failure
        if not news_headlines:
            log.info("No news headlines provided for analysis. Returning Neutral sentiment.")
            return 0 # Neutral if no news

        total_score = 0.0
        analyzed_count = 0
        try:
            # Analyze headlines. Using truncation handles headlines longer than model's max input size.
            # Batching happens automatically within the pipeline for efficiency.
            results = self.sentiment_analyzer(news_headlines, truncation=True, max_length=512)

            # Process results from the pipeline
            for result in results:
                label = result.get("label")
                # Get score, default to 0.0 if missing (shouldn't happen with standard models)
                score = result.get("score", 0.0)

                # Accumulate score based on label (POSITIVE adds, NEGATIVE subtracts)
                if label == "POSITIVE":
                    total_score += score
                elif label == "NEGATIVE":
                    total_score -= score
                # Note: Some models might output NEUTRAL labels, which are ignored here.
                analyzed_count += 1

            if analyzed_count == 0:
                 log.warning("Sentiment analysis returned no valid results for the provided headlines.")
                 return 0 # Treat as Neutral if analysis yielded nothing

            # Calculate average score across all analyzed headlines
            average_score = total_score / analyzed_count

            # Determine final sentiment signal based on average score thresholds
            # These thresholds (0.3, -0.3) can be tuned based on model and desired sensitivity.
            if average_score > 0.3:
                sentiment = 1 # Positive
            elif average_score < -0.3:
                sentiment = -1 # Negative
            else:
                sentiment = 0 # Neutral / Mixed

            log.info(f"NewsBot analysis: Headlines={len(news_headlines)}, AvgScore={average_score:.3f}, FinalSentiment={sentiment}")
            return sentiment

        except Exception as e:
            # Catch potential errors during the analysis pipeline execution
            log.error(f"Error during news sentiment analysis pipeline: {e}", exc_info=True)
            return 0 # Return neutral sentiment as a fallback on error

# Example Usage (for testing when running this file directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Setup logging for test run
    test_headlines = [
        "Stock surges on strong earnings report!",
        "Company faces new regulatory hurdles.",
        "Market remains flat amid uncertainty.",
        "Analysts upgrade stock to 'Buy'.",
        "CEO steps down unexpectedly, shares tumble.",
        "New product launch receives rave reviews."
    ]
    news_bot_test = NewsBot()
    # Check if the model loaded correctly before analyzing
    if news_bot_test.sentiment_analyzer:
        overall_sentiment = news_bot_test.analyze(test_headlines)
        if overall_sentiment is not None:
            print(f"\nOverall sentiment for test headlines: {overall_sentiment} (1=Pos, -1=Neg, 0=Neu)")
        else:
            print("\nSentiment analysis failed (model might not have loaded).")
    else:
        print("\nNewsBot model could not be loaded. Cannot perform analysis.")


