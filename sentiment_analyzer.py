from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """Performs sentiment analysis using the VADER model."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text):
        """Return sentiment category: positive, negative, or neutral."""
        if not text or not text.strip():
            return 'neutral'

        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def get_scores(self, text):
        """Return full sentiment score dictionary."""
        if not text or not text.strip():
            return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
        return self.analyzer.polarity_scores(text)

    def sentiment_icon(self, sentiment):
        """Optional emoji icon for better display."""
        icons = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}
        return icons.get(sentiment, '')
