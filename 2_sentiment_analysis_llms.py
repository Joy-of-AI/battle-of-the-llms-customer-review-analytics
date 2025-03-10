"""
Sentiment Analysis with Multiple LLMs

This script performs sentiment analysis using various Large Language Models (LLMs) 
and compares their performance across different datasets.

Features:
- Supports multiple LLMs for sentiment classification.
- Evaluates model performance using various metrics.
- Outputs structured results for easy comparison.

Author: Amir Amin
Version: 1.0
Last Updated: 2025-03-08
"""

import os
import time
import pandas as pd
import random
import re
import asyncio
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from aiolimiter import AsyncLimiter
from mistralai import Mistral
import anthropic
from llamaapi import LlamaAPI
from openai import OpenAI
import google.generativeai as genai
import tiktoken
import textwrap
from dotenv import load_dotenv

# Store your API keys for OpenAI, Anthropic, Google, DeepSeek, LLAMA, Mistral, and XAI in a .env file in the project, as below format:
# OPENAI_API_KEY = "<Your API Key for OpenAI>"
# ANTHROPIC_API_KEY = "<Your API Key for Anthropic>"
# GOOGLE_API_KEY = "<Your API Key for Google Gemini>"
# DEEPSEEK_API_KEY = "<Your API Key for DeepSeek>"
# LLAMA_API_KEY = "<Your API Key for LLAMA>"
# MISTRAL_API_KEY = "<Your API Key for Mistral>"
# XAI_API_KEY = "<Your API Key for XAI>"

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
API_KEYS = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY"),
    "LLAMA_API_KEY": os.environ.get("LLAMA_API_KEY"),
    "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
    "XAI_API_KEY": os.environ.get("XAI_API_KEY")
}

# Verify the Keys are Loaded
if not all(API_KEYS.values()):
    raise ValueError("One or more API keys are missing. Set them in the .env file.")

# Initialize API clients
openai_client = OpenAI(api_key=API_KEYS["OPENAI_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=API_KEYS["ANTHROPIC_API_KEY"])
genai.configure(api_key=API_KEYS["GOOGLE_API_KEY"])
mistral_client = Mistral(api_key=API_KEYS["MISTRAL_API_KEY"])
deepseek_client = OpenAI(api_key=API_KEYS["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
llama = LlamaAPI(API_KEYS["LLAMA_API_KEY"])
grok_client = OpenAI(api_key=API_KEYS["XAI_API_KEY"], base_url="https://api.x.ai/v1")

# Global rate limiter for Mistral Large
mistral_rate_limiter = AsyncLimiter(max_rate=1, time_period=2)

# Tokenizer for OpenAI models
openai_tokenizer = tiktoken.encoding_for_model("gpt-4")

# Function to count tokens for OpenAI models
def count_tokens(text):
    return len(openai_tokenizer.encode(text))

# Function to calculate cost based on input and output tokens
def calculate_cost(model_name, input_tokens, output_tokens):
    pricing = {
        "OpenAI GPT-4o": {"input": 2.50, "output": 10.00},
        "Claude 3.7 Sonnet": {"input": 3.00, "output": 15.00},
        "Gemini 2.0 Flash": {"input": 0.10, "output": 0.40},
        "DeepSeek V3": {"input": 0.27, "output": 1.10},
        "LLaMA 3.3 70B": {"input": 2.80, "output": 2.80},
        "Mistral Large": {"input": 2.00, "output": 6.00},
        "Grok 2": {"input": 2.00, "output": 10.00}
    }
    input_price = pricing[model_name]["input"] / 1_000_000  # Convert to cost per token
    output_price = pricing[model_name]["output"] / 1_000_000  # Convert to cost per token
    total_cost = (input_tokens * input_price) + (output_tokens * output_price)
    return round(total_cost, 6)

# Function to validate sentiment output
def validate_sentiment(output):
    valid_sentiments = {"positive", "negative", "neutral"}
    output = output.lower().strip(".,!?")
    for sentiment in valid_sentiments:
        if sentiment in output:
            return sentiment
    return "error"

# Function to add noise to text
def add_noise(text, noise_level=0.1):
    noisy_text = list(text)
    num_noise_chars = int(noise_level * len(text))  # Number of characters to replace
    for _ in range(num_noise_chars):
        pos = random.randint(0, len(text) - 1)
        noisy_text[pos] = random.choice(["x", "y", "z", " ", "123"])  # Example noise
    return "".join(noisy_text)

# Function to wrap a text
def wrap_text(df, column_name, width=50):
    df[column_name] = df[column_name].apply(lambda x: "\n".join(textwrap.wrap(str(x), width)))
    return df
    
# Sentiment analysis function
async def analyze_sentiment_with_time_cost(text, model_name, max_tokens=10, temperature=0):
    try:
        prompt = f"""
        Analyze the sentiment of the following text and return only 'positive', 'negative', or 'neutral'. 
        Focus on the tone and key phrases in the text. Here are some examples:
        1. "I love this product! It works perfectly." → Positive
        2. "The service was terrible and the staff was rude." → Negative
        3. "The product arrived on time." → Neutral

        Now analyze this text: {text}
        """
        input_tokens = count_tokens(prompt)

        if model_name == "Mistral Large":
            async with mistral_rate_limiter:
                start_time = time.time()
                response = mistral_client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                end_time = time.time()
                output = response.choices[0].message.content.strip().lower()
                output_tokens = len(openai_tokenizer.encode(output))
                api_response_time = end_time - start_time
        else:
            start_time = time.time()
            if model_name == "OpenAI GPT-4o":
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content.strip().lower()
                output_tokens = response.usage.completion_tokens
            elif model_name == "Claude 3.7 Sonnet":
                response = anthropic_client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                output = response.content[0].text.strip().lower()
                output_tokens = len(openai_tokenizer.encode(response.content[0].text))
            elif model_name == "Gemini 2.0 Flash":
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
                output = response.text.strip().lower()
                output_tokens = len(openai_tokenizer.encode(output))
            elif model_name == "DeepSeek V3":
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content.strip().lower()
                output_tokens = response.usage.completion_tokens
            elif model_name == "LLaMA 3.3 70B":
                api_request_json = {
                    "model": "llama3.3-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
                response = llama.run(api_request_json)
                output = response.json()["choices"][0]["message"]["content"].strip().lower()
                output_tokens = len(openai_tokenizer.encode(output))
            elif model_name == "Grok 2":
                response = grok_client.chat.completions.create(
                    model="grok-2-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content.strip().lower()
                output_tokens = response.usage.completion_tokens
            else:
                return "Unsupported model", 0, 0

            end_time = time.time()
            api_response_time = end_time - start_time

        output = validate_sentiment(output)
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        return output, api_response_time, cost

    except Exception as e:
        return str(e), 0, 0

# Function to calculate performance metrics
def calculate_metrics(true_sentiments, predicted_sentiments):
    accuracy = accuracy_score(true_sentiments, predicted_sentiments)
    f1 = f1_score(true_sentiments, predicted_sentiments, average="weighted")
    recall = recall_score(true_sentiments, predicted_sentiments, average="weighted")
    precision = precision_score(true_sentiments, predicted_sentiments, average="weighted")
    return accuracy, f1, recall, precision

# Function to generate and display tables
async def generate_tables(data_type, reviews, true_sentiments, models):
    results = []
    for i, review in enumerate(reviews):
        row = {"Review": review, "True Sentiment": true_sentiments[i]}
        for model_name, func in models.items():
            sentiment, response_time, cost = await func(review, model_name)
            row[model_name] = sentiment
            row[f"{model_name}_response_time"] = response_time
            row[f"{model_name}_cost"] = cost
        results.append(row)
   
    table1_df = pd.DataFrame(results)

    # Convert results to DataFrame for Table 1
    table1_df["Review"] = table1_df["Review"].apply(lambda x: "\n".join(textwrap.wrap(str(x), width=50)))

    print(f"\nSentiment Analysis- {data_type} Results")
    print(tabulate(table1_df[["Review", "True Sentiment"] + list(models.keys())], 
                   headers="keys", 
                   tablefmt="grid", 
                   stralign="left",
                   showindex=False))

    # Calculate performance metrics for Table 2
    performance_metrics = []
    for model_name in models.keys():
        true_sentiments = [row["True Sentiment"] for row in results]
        predicted_sentiments = [row[model_name] for row in results]
        
        accuracy, f1, recall, precision = calculate_metrics(true_sentiments, predicted_sentiments)
        total_response_time = sum(row[f"{model_name}_response_time"] for row in results)
        total_cost = sum(row[f"{model_name}_cost"] for row in results)
        
        performance_metrics.append({
            "LLM": model_name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Recall": recall,
            "Precision": precision,
            "Total Response Time": total_response_time,
            "Total Cost": total_cost
        })

    # Convert performance metrics to DataFrame
    table2_df = pd.DataFrame(performance_metrics)
    print(f"\nPerformance Metrics for {data_type} Data")
    print(tabulate(table2_df, 
                   headers="keys", 
                   tablefmt="grid",
                   stralign="left",
                   floatfmt=".5f", 
                   showindex=False))

# Main function to run the analysis
async def main():
    # Original reviews and true sentiments
    original_reviews = final_sample["review"]
    true_sentiments = ["negative", "negative", "negative", "neutral", "negative", "positive", "negative", "positive", "negative", "neutral"]

    # Edge cases and their true sentiments
    edge_cases = [
        "I love the product, but the delivery was terrible.",  # Mixed sentiment
        "This is the worst experience ever!!!",  # Strong negative sentiment
        "Meh, it's okay I guess.",  # Neutral sentiment
        "Wow, just wow. Amazing!",  # Strong positive sentiment
        "I can't even...",  # Ambiguous sentiment
        "The product is good, but the service is bad.",  # Mixed sentiment
    ]
    edge_case_true_sentiment = ["neutral", "negative", "neutral", "positive", "neutral", "neutral"]

    # Models to analyze
    models = {
        "OpenAI GPT-4o": analyze_sentiment_with_time_cost,
        "Claude 3.7 Sonnet": analyze_sentiment_with_time_cost,
        "Gemini 2.0 Flash": analyze_sentiment_with_time_cost,
        "LLaMA 3.3 70B": analyze_sentiment_with_time_cost,
        "Mistral Large": analyze_sentiment_with_time_cost,
        "DeepSeek V3": analyze_sentiment_with_time_cost,
        "Grok 2": analyze_sentiment_with_time_cost
    }

    # Analyze original reviews
    await generate_tables("Customer Reviews", original_reviews, true_sentiments, models)

    # Analyze edge cases
    await generate_tables("Edge Cases", edge_cases, edge_case_true_sentiment, models)

    # Add noise to original reviews and analyze
    noisy_reviews = [add_noise(review, noise_level=0.1) for review in original_reviews]
    await generate_tables("Customer Reviews with Noisy", noisy_reviews, true_sentiments, models)
   
# Run the analysis
if __name__ == "__main__":
    asyncio.run(main())
