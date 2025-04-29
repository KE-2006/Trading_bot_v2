# TradingRobotTeamV2/agents/news_bot.py
# Using the same NewsBot as before, as Transformers are already quite good.
# No major changes needed here for this version upgrade.

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Optional
import logging
import torch # Explicitly import torch to check availability

log = logging.getLogger(__name__)

class NewsBot:
    """
    Analyzes the sentiment of news headlines using a pre-trained transformer model.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the NewsBot by loading the sentiment analysis model.

        Args:
            model_name: The name of the Hugging Face model to use.
        """
        self.model_name = model_name
        self.sentiment_analyzer = None
        self.device = self._get_device() # Determine if GPU is available
        self._load_model()

    def _get_device(self):
        """Checks if CUDA (GPU) is available, otherwise uses CPU."""
        if torch.cuda.is_available():
            log.info("CUDA (GPU) available, NewsBot will use GPU.")
            return 0 # Device ID 0 for CUDA
        else:
            log.info("CUDA not available, NewsBot will use CPU.")
            return -1 # Device ID -1 for CPU

    def _load_model(self):
        """Loads the sentiment analysis pipeline."""
        try:
            log.info(f"Loading sentiment analysis model: {self.model_name}...")
            # Load tokenizer and model explicitly
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            # Create pipeline and assign device
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=self.device # Use GPU if available, else CPU
            )
            log.info(f"Sentiment analysis model loaded successfully onto {'GPU' if self.device == 0 else 'CPU'}.")
        except Exception as e:
            log.error(f"Failed to load sentiment analysis model '{self.model_name}': {e}", exc_info=True)
            self.sentiment_analyzer = None

    def analyze(self, news_headlines: List[str]) -> Optional[int]:
        """
        Analyzes a list of news headlines and returns an overall sentiment score.

        Args:
            news_headlines: A list of strings, where each string is a news headline.

        Returns:
            1 for overall positive sentiment.
           -1 for overall negative sentiment.
            0 for neutral or mixed sentiment (or if analysis fails).
           None if the model wasn't loaded.
        """
        if not self.sentiment_analyzer:
            log.error("Sentiment analyzer model not loaded. Cannot analyze news.")
            return None
        if not news_headlines:
            log.info("No news headlines provided for analysis.")
            return 0 # Neutral if no news

        total_score = 0.0
        analyzed_count = 0
        try:
            # Analyze headlines in batch
            # Truncation=True handles headlines longer than the model's max input size
            results = self.sentiment_analyzer(news_headlines, truncation=True, max_length=512)

            for result in results:
                label = result.get("label")
                score = result.get("score", 0.0)

                if label == "POSITIVE":
                    total_score += score
                elif label == "NEGATIVE":
                    total_score -= score
                analyzed_count += 1

            if analyzed_count == 0:
                 log.warning("Sentiment analysis returned no valid results.")
                 return 0

            average_score = total_score / analyzed_count

            # Determine final sentiment signal (thresholds can be adjusted)
            if average_score > 0.3: sentiment = 1
            elif average_score < -0.3: sentiment = -1
            else: sentiment = 0

            log.info(f"NewsBot analysis: Headlines={len(news_headlines)}, AvgScore={average_score:.3f}, Sentiment={sentiment}")
            return sentiment

        except Exception as e:
            log.error(f"Error during news sentiment analysis: {e}", exc_info=True)
            return 0 # Return neutral on error

# Example Usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    news = [
        "Stock surges on strong earnings report!",
        "Company faces new regulatory hurdles.",
        "Market remains flat amid uncertainty.",
        "Analysts upgrade stock to 'Buy'.",
        "CEO steps down unexpectedly."
    ]
    bot = NewsBot()
    if bot.sentiment_analyzer:
        sentiment = bot.analyze(news)
        if sentiment is not None:
            print(f"Overall sentiment for news list: {sentiment} (1=Pos, -1=Neg, 0=Neu)")
        else:
            print("Sentiment analysis failed.")
    else:
        print("NewsBot model could not be loaded.")
