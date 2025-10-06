"""
Quick test script to verify both sentiment analysis and text generation.
Run before launching Streamlit to ensure all components are functional.
"""

from sentiment_analyzer import SentimentAnalyzer
from text_generator import TextGenerator

# Initialize models
analyzer = SentimentAnalyzer()
generator = TextGenerator()

print("=== Sentiment Analysis Test ===\n")

test_texts = [
    "I love this amazing day!",
    "This is terrible news",
    "The meeting is at 3pm",
    "",
    "The food was okay, nothing special."
]

for text in test_texts:
    sentiment = analyzer.analyze(text)
    scores = analyzer.get_scores(text)
    print(f"Text: '{text}'")
    print(f" → Sentiment: {sentiment.upper()} (Compound: {scores['compound']:.2f})")
    print("-" * 50)

print("\n=== Text Generation Test ===\n")

prompt = "a day at the park"
for sentiment in ['positive', 'negative', 'neutral']:
    print(f"\n--- {sentiment.upper()} TEXT ---")
    text = generator.generate_text(prompt, sentiment, 80)
    print(text)
    print("-" * 80)

print("\n✅ Tests completed successfully!")
