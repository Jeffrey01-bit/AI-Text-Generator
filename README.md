# AI Text Generator

An AI-powered text generator that creates sentiment-aligned essays and paragraphs based on input prompts.

## Features

- **Sentiment Analysis**: Automatically detects sentiment (positive, negative, neutral) from input prompts with emoji indicators
- **Text Generation**: Generates coherent essays/paragraphs aligned with detected or manually selected sentiment
- **Interactive Interface**: User-friendly Streamlit frontend with enhanced UI elements
- **Customizable Output**: Adjustable text length (50-300 words) and manual sentiment override
- **GPU Support**: Automatic GPU acceleration when available for faster generation
- **Smart Validation**: Input validation and error handling for robust performance

## Technical Approach

### Sentiment Analysis
- **VADER Sentiment Analyzer**: Lexicon-based tool optimized for social media text
- **Real-time Analysis**: No training required, works out-of-the-box
- **Compound Scoring**: Uses compound score thresholds (>0.05 positive, <-0.05 negative)
- **Visual Feedback**: Emoji indicators for sentiment categories
- **Input Validation**: Handles empty or invalid inputs gracefully

### Text Generation
- **GPT-2 Model**: Pre-trained transformer model for text generation
- **Sentiment Starters**: Uses randomized sentiment-specific opening phrases for natural flow
- **Advanced Parameters**: Temperature 0.7, top-p 0.9, repetition penalty 1.2 for quality output
- **Smart Length Control**: Token-based length targeting for accurate word counts
- **Clean Output**: Automatic sentence completion and proper punctuation

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access Interface**: Open browser to `http://localhost:8501`

## Usage

1. Enter a prompt in the text area (e.g., "A day at the beach")
2. Review auto-detected sentiment with emoji indicator or manually override
3. Adjust text length using the slider (50-300 words)
4. Click "Generate Text" to create sentiment-aligned essays/paragraphs
5. View sentiment scores and download generated content

## Project Structure

```
AI Text Generator/
├── app.py                 # Streamlit frontend
├── sentiment_analyzer.py  # VADER sentiment analysis
├── text_generator.py      # GPT-2 text generation
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Challenges & Solutions

1. **Model Loading Time**: Used `@st.cache_resource` to cache models and GPU acceleration
2. **Sentiment Alignment**: Implemented randomized sentiment-specific starter phrases for natural flow
3. **Output Quality**: Optimized generation parameters (temperature 0.7, top-p 0.9, repetition penalty 1.2)
4. **Text Coherence**: Smart sentence completion and proper punctuation handling
5. **User Experience**: Added emoji indicators, input validation, and enhanced UI feedback
6. **Performance**: GPU support for faster generation when available

## Current Enhancements

- ✅ **Export Functionality**: Download generated text as .txt files
- ✅ **GPU Acceleration**: Automatic GPU detection and usage
- ✅ **Enhanced UI**: Emoji indicators and better visual feedback
- ✅ **Smart Generation**: Randomized starters and improved text quality
- ✅ **Input Validation**: Robust error handling and edge case management

## Future Enhancements

- Deploy to Streamlit Cloud for public access
- Add more sophisticated sentiment models (BERT, RoBERTa)
- Implement custom fine-tuned generation models
- Add multiple output format options (essay, story, poem)
- Include text quality metrics and readability scores

## Dependencies

- `streamlit`: Web interface framework
- `transformers`: Hugging Face transformers for GPT-2
- `torch`: PyTorch backend
- `vaderSentiment`: Sentiment analysis tool