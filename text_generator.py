from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

class TextGenerator:
    """Generates sentiment-aligned text using GPT-2."""

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sentiment_starters = {
            'positive': [
                "It was an amazing experience when",
                "I felt so happy about", 
                "The wonderful thing about",
                "What a delightful day it was when"
            ],
            'negative': [
                "It was frustrating when",
                "I was really annoyed by",
                "The worst part about", 
                "I couldn't believe how terrible"
            ],
            'neutral': [
                "Today I experienced",
                "Let me tell you about",
                "Here's what happened with",
                "I want to describe"
            ]
        }

    def generate_text(self, prompt, sentiment='neutral', max_length=150):
        """Generate text based on sentiment and prompt."""
        if not prompt or not prompt.strip():
            return "Please enter a valid prompt."

        sentiment = sentiment.lower()
        if sentiment not in self.sentiment_starters:
            sentiment = 'neutral'

        starter = random.choice(self.sentiment_starters[sentiment])
        full_prompt = f"{starter} {prompt}. "

        inputs = self.tokenizer.encode(full_prompt, return_tensors='pt')

        target_tokens = max_length // 3

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + target_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text.strip()

        # Ensure clean ending
        if not result.endswith(('.', '!', '?')):
            last_punct = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
            if last_punct > len(full_prompt):
                result = result[:last_punct + 1]
            else:
                result += '.'

        return result
