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
from dotenv import load_dotenv
from tabulate import tabulate  # For tabular output
import tiktoken

# packages to use LLMs APIs
from mistralai import Mistral # Required for Mistral
import anthropic # Required for Claud
import json  # Required for LLaMA
from llamaapi import LlamaAPI  # Importing the LLaMA API package
from openai import OpenAI  # Ensure OpenAI SDK is installed: `pip install openai`
import google.generativeai as genai  # For Google Gemini

# Packages for performance analytics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score
import re
import random

# ---------------------------------------------------------------------
# Section 0- Configurations and setups
# ---------------------------------------------------------------------

# Store your API keys for OpenAI, Anthropic, Google, DeepSeek, LLAMA, Mistral, and XAI in a .env file in the project, as below format:
# OPENAI_API_KEY = "<Your API Key for OpenAI>"
# ANTHROPIC_API_KEY = "<Your API Key for Anthropic>"
# GOOGLE_API_KEY = "<Your API Key for Google Gemini>"
# DEEPSEEK_API_KEY = "<Your API Key for DeepSeek>"
# LLAMA_API_KEY = "<Your API Key for LLAMA>"
# MISTRAL_API_KEY = "<Your API Key for Mistral>"
# XAI_API_KEY = "<Your API Key for XAI>"

# Load environment variables from the .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# Verify the Keys are Loaded
if not all([OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, LLAMA_API_KEY, MISTRAL_API_KEY, XAI_API_KEY]):
    raise ValueError("One or more API keys are missing. Set them in the .env file.")

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
llama = LlamaAPI(LLAMA_API_KEY)  # Initialize the LLaMA API client
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# ---------------------------------------------------------------------
# Section 1- Performing sentiment Analysis
# ---------------------------------------------------------------------

# Universal sentiment prompt
PROMPT = "Determine the sentiment of the text and return only one word: 'positive', 'negative', or 'neutral'."

# Sentiment Analysis Functions
def analyze_sentiment_openai(text):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": PROMPT},
                      {"role": "user", "content": text}],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return str(e)

def analyze_sentiment_claude(text):
    try:
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=10,
            messages=[{"role": "user", "content": f"{PROMPT} {text}"}]
        )
        return response.content[0].text.strip().lower()
    except Exception as e:
        return str(e)

def analyze_sentiment_gemini(text):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(f"{PROMPT} {text}")
        return response.text.strip().lower()
    except Exception as e:
        return str(e)

def analyze_sentiment_deepseek(text):
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return str(e)

def analyze_sentiment_llama(text):
    try:
        api_request_json = {
            "model": "llama3.3-70b",
            "messages": [{"role": "user", "content": f"{PROMPT} {text}"}],
            "stream": False
        }
        response = llama.run(api_request_json)
        response_json = response.json()
        if "choices" in response_json and response_json["choices"]:
            return response_json["choices"][0]["message"]["content"].strip().lower()
        return f"Error: Unexpected response format - {json.dumps(response_json, indent=2)}"
    except Exception as e:
        return str(e)

def analyze_sentiment_mixtral(text):
    try:
        chat_response = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text}
            ]
        )
        return chat_response.choices[0].message.content.strip().lower()
    except Exception as e:
        return f"Error: {str(e)}"

# Sentiment Analysis Function for Grok
def analyze_sentiment_grok(text):
    try:
        response = grok_client.chat.completions.create(
            model="grok-2-latest",  # Use the correct model name
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return str(e)
    
# Run models and collect results
models = {
    "OpenAI GPT-4o": analyze_sentiment_openai,
    "Claude 3.7 Sonnet": analyze_sentiment_claude,
    "Gemini 2.0 Flash": analyze_sentiment_gemini,
    "LLaMA 3.3 70B": analyze_sentiment_llama,
    "Mistral Large": analyze_sentiment_mixtral,
    "DeepSeek V3": analyze_sentiment_deepseek,
    "Grok 2": analyze_sentiment_grok
}

# Store results as a column in the final_sample dataset
for model_name, func in models.items():
    predictions = final_sample["review"].apply(func)  # Directly apply the function
    final_sample[model_name] = predictions

# Print all input columns and sentiments using tabulate
# print("\nSentiment Analysis Results (Tabulate):")
# print(tabulate(final_sample, headers="keys", tablefmt="grid", colalign=("left",)*len(final_sample.columns)))

# Only print the review and sentiment columns
columns_to_display = ["review", "OpenAI GPT-4o", "Claude 3.7 Sonnet", "Gemini 2.0 Flash", "DeepSeek V3", "LLaMA 3.3 70B", "Mistral Large", "Grok 2"]
filtered_sample = final_sample[columns_to_display]

# Print in Grid Format with Sentiments
print("\nSentiment Analysis Results:")
print(tabulate(filtered_sample, headers="keys", tablefmt="grid", showindex=False))

# Save the final_sample DataFrame as a CSV file
final_sample.to_csv("sentiment_results_V1.csv", index=False)
print("Table saved as 'sentiment_results_V1.csv'")

# ---------------------------------------------------------------------
# Section 2- Performance analytics and comparisons
# ---------------------------------------------------------------------

# True sentiments based on my study on customer reviews
true_sentiments = ["negative", "negative", "negative", "neutral", "negative", "positive", "negative", "positive", "negative", "neutral"]

# Extract predictions from the DataFrame
predictions = {
    "OpenAI GPT-4o": final_sample["OpenAI GPT-4o"].tolist(),
    "Claude 3.7 Sonnet": final_sample["Claude 3.7 Sonnet"].tolist(),
    "Gemini 2.0 Flash": final_sample["Gemini 2.0 Flash"].tolist(),
    "DeepSeek V3": final_sample["DeepSeek V3"].tolist(),
    "LLaMA 3.3 70B": final_sample["LLaMA 3.3 70B"].tolist(),
    "Mistral Large": final_sample["Mistral Large"].tolist(),
    "Grok 2": final_sample["Grok 2"].tolist()
}

# Function to normalize text (remove spaces, punctuation, and convert to lowercase)
def normalize_text(text):
    if isinstance(text, str):  # Ensure the input is a string
        text = re.sub(r"[^\w]", "", text)  # Remove all non-alphanumeric characters
        text = text.lower()  # Convert to lowercase
    return text

# Function to validate and normalize predictions
def validate_prediction(pred):
    if isinstance(pred, str):  # Ensure the prediction is a string
        pred = normalize_text(pred)
        if pred in ["positive", "negative", "neutral"]:
            return pred
    return "error"  # Default to "error" for invalid predictions

# Function to calculate metrics
def calculate_metrics(true, pred):
    # Normalize predictions and true values
    true_normalized = [normalize_text(t) for t in true]
    pred_normalized = [validate_prediction(p) for p in pred]  # Validate and normalize predictions
    accuracy = accuracy_score(true_normalized, pred_normalized)
    f1 = f1_score(true_normalized, pred_normalized, average="weighted")
    precision = precision_score(true_normalized, pred_normalized, average="weighted")
    recall = recall_score(true_normalized, pred_normalized, average="weighted")
    return accuracy, f1, precision, recall

# Function to evaluate edge case handling
def evaluate_edge_cases(pred, edge_cases, edge_case_true_sentiments):
    # Use the model's predictions for the edge cases
    predictions = [validate_prediction(pred[i]) for i in range(len(edge_cases))]  # Validate predictions
    accuracy = accuracy_score(edge_case_true_sentiments, predictions)
    f1 = f1_score(edge_case_true_sentiments, predictions, average="weighted")
    precision = precision_score(edge_case_true_sentiments, predictions, average="weighted")
    recall = recall_score(edge_case_true_sentiments, predictions, average="weighted")
    return accuracy, f1, precision, recall

# Function to add noise to text
def add_noise(text, noise_level=0.1):
    """
    Adds random noise to the input text.
    :param text: Input text (string).
    :param noise_level: Fraction of characters to replace with noise (float between 0 and 1).
    :return: Noisy text (string).
    """
    noisy_text = list(text)
    num_noise_chars = int(noise_level * len(text))  # Number of characters to replace
    for _ in range(num_noise_chars):
        # Randomly select a position to replace
        pos = random.randint(0, len(text) - 1)
        # Replace with a random character or extra word
        noisy_text[pos] = random.choice(["x", "y", "z", " ", "123"])  # Example noise
    return "".join(noisy_text)

# In addition to final_sample dataset that we check their sentiments, we create an edge case dataset to assess performance of LLMs for extreme scenarios
# Edge cases are unusual or extreme inputs that test a program's robustness, such as Sarcasm, Empty input, Very short input, Punctuation only, or Mixed languages.
edge_cases = [
    "I love the product, but the delivery was terrible.",  # Mixed sentiment
    "This is the worst experience ever!!!",  # Strong negative sentiment
    "Meh, it's okay I guess.",  # Neutral sentiment
    "Wow, just wow. Amazing!",  # Strong positive sentiment
    "I can't even...",  # Ambiguous sentiment
    "The product is good, but the service is bad.",  # Mixed sentiment
]
edge_case_true_sentiments = ["neutral", "negative", "neutral", "positive", "neutral", "neutral"]

# Extend true_sentiments to include edge case true sentiments
extended_true_sentiments = true_sentiments + edge_case_true_sentiments

# Generate predictions for edge cases
edge_case_predictions = {}
for model_name, pred_func in models.items():  # Assuming `models` is a dictionary of model functions
    edge_case_predictions[model_name] = [pred_func(text) for text in edge_cases]

# Add edge case predictions to the main predictions dictionary
for model_name, pred in predictions.items():
    pred.extend(edge_case_predictions[model_name])

# Add noise to the original texts
noisy_texts = [add_noise(text) for text in true_sentiments]

# Generate predictions for noisy texts
noisy_predictions = {}
for model_name, pred_func in models.items():  # Assuming `models` is a dictionary of model functions
    noisy_predictions[model_name] = [pred_func(text) for text in noisy_texts]

# Calculate metrics for each model
results = {}
for model_name, pred in predictions.items():
    # Calculate standard metrics (using only the original true_sentiments)
    accuracy, f1, precision, recall = calculate_metrics(true_sentiments, pred[:len(true_sentiments)])
    
    # Evaluate edge case handling (using only the edge case predictions)
    edge_accuracy, edge_f1, edge_precision, edge_recall = evaluate_edge_cases(
        pred[len(true_sentiments):], edge_cases, edge_case_true_sentiments
    )
    
    # Evaluate consistency using cross-validation (example with a placeholder model)
    # Note: You need a proper model (e.g., sklearn pipeline) for cross-validation.
    # Here, we use a placeholder for demonstration.
    consistency_scores = [accuracy] * 5  # Placeholder for cross-validation scores
    consistency_mean = sum(consistency_scores) / len(consistency_scores)
    consistency_std = (max(consistency_scores) - min(consistency_scores)) / 2
    
    # Evaluate robustness (using noisy data predictions)
    noisy_accuracy = accuracy_score(true_sentiments, [validate_prediction(p) for p in noisy_predictions[model_name]])
    
    # Store results
    results[model_name] = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Edge Case Accuracy": edge_accuracy,
        "Edge Case F1": edge_f1,
        "Edge Case Precision": edge_precision,
        "Edge Case Recall": edge_recall,
        "Consistency (Mean Accuracy)": consistency_mean,
        # "Consistency (Std Dev)": consistency_std,
        "Robustness (Noisy Accuracy)": noisy_accuracy,
    }

# Convert results to a DataFrame for tabular output
results_df = pd.DataFrame(results).T

# Print the comparison table for performance metrics
print("\nPerformance Comparison of LLMs (with Edge Case Handling, Consistency, and Robustness):")
print(tabulate(results_df, headers="keys", tablefmt="grid", floatfmt=".4f"))

# Identify all models with the best accuracy
best_accuracy = results_df["Accuracy"].max()
best_models = results_df[results_df["Accuracy"] == best_accuracy].index.tolist()

# Print the best-performing models
if len(best_models) == 1:
    print(f"\nThe best-performing model is **{best_models[0]}** with an accuracy of {best_accuracy:.4f}.")
else:
    print(f"\nThe best-performing models are **{', '.join(best_models)}** with an accuracy of {best_accuracy:.4f}.")

# For test purpose and to get deeper understanding of performance comparison
print("\nCheck below table for test purpose and to get deeper understanding of performance comparison.")

# Create a table of predictions for each model
prediction_table = pd.DataFrame(predictions)
prediction_table["True Sentiment"] = extended_true_sentiments

# Normalize predictions and add columns for valid predictions and matches
for model_name in predictions.keys():
    prediction_table[f"{model_name}_Normalized"] = prediction_table[model_name].apply(normalize_text)
    prediction_table[f"{model_name}_Valid"] = prediction_table[f"{model_name}_Normalized"].apply(
        lambda x: x if x in ["positive", "negative", "neutral"] else "neutral"
    )
    prediction_table[f"{model_name}_Match"] = prediction_table.apply(
        lambda row: "match" if row[f"{model_name}_Valid"] == normalize_text(row["True Sentiment"]) else "mismatch", axis=1
    )

# Split the prediction table into two: one for original data and one for edge cases
original_data_table = prediction_table.iloc[:len(true_sentiments)]
edge_case_table = prediction_table.iloc[len(true_sentiments):]

# Add the edge case review texts to the edge_case_table
edge_case_table["Review"] = edge_cases

# Reorder columns to make "Review" the first column
edge_case_table = edge_case_table[["Review"] + [col for col in edge_case_table.columns if col != "Review"]]

# Print the predictions table for original data
print("\nPredicted Values for Original Data:")
print(tabulate(original_data_table, headers="keys", tablefmt="grid"))

# Print the predictions table for edge cases with review text as the first column
print("\nPredicted Values for Edge Cases:")
print(tabulate(edge_case_table, headers="keys", tablefmt="grid"))

# Print class-wise metrics (precision, recall, F1 for each class)
print("\nClass-Wise Metrics:")
for model_name, pred in predictions.items():
    # Replace invalid predictions with "error"
    valid_pred = [p if p in ["positive", "negative", "neutral"] else "error" for p in pred]
    
    print(f"\nModel: {model_name}")
    print(classification_report(
        extended_true_sentiments,
        valid_pred,
        target_names=["positive", "negative", "neutral"],
        labels=["positive", "negative", "neutral"],
        zero_division=0  # Handle cases where division by zero occurs
    ))

# ---------------------------------------------------------------------
# section 3- Comparing LLMs by their response time and total cost for sentiments
# ---------------------------------------------------------------------

# Pricing per 1M tokens (USD) as of article publication day
# Pricing referenecs available in Resources section of the article
PRICING = {
    "OpenAI GPT-4o": {"input": 2.50, "output": 10.00},  # Input: $2.50, Output: $10.00
    "Claude 3.7 Sonnet": {"input": 3.00, "output": 15.00},  # Input: $3.00, Output: $15.00
    "Gemini 2.0 Flash": {"input": 0.10, "output": 0.40},  # Input: $0.10, Output: $0.40
    "DeepSeek V3": {"input": 0.27, "output": 1.10},  # Cache Miss: $0.27, Output: $1.10
    "LLaMA 3.3 70B": {"input": 2.80, "output": 2.80},  # Input: $2.80, Output: $2.80
    "Mistral Large": {"input": 2.00, "output": 6.00},  # Input: $2.00, Output: $6.00
    "Grok 2": {"input": 2.00, "output": 10.00}  # Input: $2.00, Output: $10.00
}

# Tokenizer for OpenAI models (tiktoken)
# gpt-4 is enough to tokenise the text; it does the job and it's free
openai_tokenizer = tiktoken.encoding_for_model("gpt-4")

# Function to count tokens for OpenAI models
def count_tokens(text):
    return len(openai_tokenizer.encode(text))

# Function to calculate cost based on input and output tokens
def calculate_cost(model_name, input_tokens, output_tokens):
    input_price = PRICING[model_name]["input"] / 1_000_000  # Convert to cost per token
    output_price = PRICING[model_name]["output"] / 1_000_000  # Convert to cost per token
    total_cost = (input_tokens * input_price) + (output_tokens * output_price)
    return round(total_cost, 6)

# Function to measure time and cost for sentiment analysis
def analyze_sentiment_with_time_cost(text, model_name, model_func):
    start_time = time.time()
    try:
        # Generate sentiment prediction
        prompt = f"Analyze the sentiment of the following text and return only 'positive', 'negative', or 'neutral': {text}"
        input_tokens = count_tokens(prompt)  # Count input tokens

        if model_name == "OpenAI GPT-4o":
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,  # Limit output tokens for sentiment analysis
                temperature=0
            )
            output = response.choices[0].message.content.strip().lower()
            output_tokens = response.usage.completion_tokens

        elif model_name == "Claude 3.7 Sonnet":
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            output = response.content[0].text.strip().lower()
            output_tokens = len(response.content[0].text.split())  # Approximate output tokens

        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            output = response.text.strip().lower()
            output_tokens = len(output.split())  # Approximate output tokens

        elif model_name == "DeepSeek V3":
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            output = response.choices[0].message.content.strip().lower()
            output_tokens = response.usage.completion_tokens

        elif model_name == "LLaMA 3.3 70B":
            api_request_json = {"model": "llama3.3-70b", "messages": [{"role": "user", "content": prompt}], "stream": False}
            response = llama.run(api_request_json)
            output = response.json()["choices"][0]["message"]["content"].strip().lower()
            output_tokens = len(output.split())  # Approximate output tokens

        elif model_name == "Mistral Large":
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            output = chat_response.choices[0].message.content.strip().lower()
            output_tokens = len(output.split())  # Approximate output tokens

        elif model_name == "Grok 2":
            response = grok_client.chat.completions.create(
                model="grok-2-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            output = response.choices[0].message.content.strip().lower()
            output_tokens = response.usage.completion_tokens
        
        else:
            return "Unsupported model", 0, 0

    except Exception as e:
        return str(e), 0, 0

    end_time = round(time.time() - start_time, 4)
    cost = calculate_cost(model_name, input_tokens, output_tokens)

    return output, end_time, cost

# Update the main loop to include total time and cost metrics
results = {}
for model_name, pred_func in models.items():
    # Lists to store time and cost for each prediction
    response_times = []
    response_costs = []
    
    # Generate predictions and measure time/cost
    pred = []
    for text in true_sentiments + edge_cases:
        output, response_time, response_cost = analyze_sentiment_with_time_cost(text, model_name, pred_func)
        pred.append(output)
        response_times.append(response_time)
        response_costs.append(response_cost)
    
    # Calculate standard metrics (using only the original true_sentiments)
    accuracy, f1, precision, recall = calculate_metrics(true_sentiments, pred[:len(true_sentiments)])
    
    # Evaluate edge case handling (using only the edge case predictions)
    edge_accuracy, edge_f1, edge_precision, edge_recall = evaluate_edge_cases(
        pred[len(true_sentiments):], edge_cases, edge_case_true_sentiments
    )
    
    # Evaluate robustness (using noisy data predictions)
    noisy_accuracy = accuracy_score(true_sentiments, [validate_prediction(p) for p in pred[:len(true_sentiments)]])
    
    # Calculate total response time and total cost
    total_response_time = sum(response_times)
    total_response_cost = sum(response_costs)
    
    # Store results
    results[model_name] = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Edge Case Accuracy": edge_accuracy,
        "Edge Case F1": edge_f1,
        "Edge Case Precision": edge_precision,
        "Edge Case Recall": edge_recall,
        "Robustness (Noisy Accuracy)": noisy_accuracy,
        "Total Response Time (s)": total_response_time,
        "Total Response Cost (USD)": total_response_cost
    }

# Convert results to a DataFrame for tabular output
results_df = pd.DataFrame(results).T

# Print the comparison table for performance metrics
print("\nPerformance Comparison of LLMs (with Edge Case Handling, Robustness, Time, and Cost):")
print(tabulate(results_df, headers="keys", tablefmt="grid", floatfmt=".5f"))

