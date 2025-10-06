import streamlit as st
from sentiment_analyzer import SentimentAnalyzer
from text_generator import TextGenerator

# Initialize components
@st.cache_resource
def load_models():
    return SentimentAnalyzer(), TextGenerator()

def main():
    st.title("🤖 AI Text Generator")
    st.write("Generate an essay or paragraph aligned with the sentiment of your prompt.")

    # Load models
    sentiment_analyzer, text_generator = load_models()

    # User input
    prompt = st.text_area("Enter your prompt:", placeholder="e.g., A day at the beach...")

    # Auto-detect sentiment
    auto_sentiment = sentiment_analyzer.analyze(prompt) if prompt else 'neutral'
    sentiment_override = st.selectbox(
        "Sentiment (auto-detected or manual):",
        ['positive', 'negative', 'neutral'],
        index=['positive', 'negative', 'neutral'].index(auto_sentiment)
    )

    # Length slider
    max_length = st.slider("Text length:", 50, 300, 150)

    if st.button("Generate Text") and prompt:
        with st.spinner("Analyzing sentiment and generating text..."):
            scores = sentiment_analyzer.get_scores(prompt)
            st.write(f"**Detected Sentiment:** {auto_sentiment.title()} {sentiment_analyzer.sentiment_icon(auto_sentiment)}")
            st.write(f"**Scores:** Positive: {scores['pos']:.2f}, Negative: {scores['neg']:.2f}, Neutral: {scores['neu']:.2f}")

            generated_text = text_generator.generate_text(prompt, sentiment_override, max_length)

            st.subheader("Generated Text:")
            st.write(generated_text)

            st.download_button(
                "Download Generated Text",
                generated_text,
                file_name="generated_text.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
