"""
End-to-End Sentiment Analysis, Summarisation, Response Generation with LLMs, 
and comparion of end-to-end response time and cost for each llm

This script performs the entire pipeline of sentiment analysis, 
summarisation, and response generation using multiple leading LLMs, 
while also comparing their total response time and cost.

Features:
- End-to-end processing from sentiment analysis to response generation.
- Comparison of multiple LLMs with respect to response time and cost.
- Outputs detailed results for each model and step in the process.
- Parallel processing using asyncio to run the tasks concurrently and improving performance.

Author: Amir Amin
Version: 1.0
Last Updated: 2025-03-08
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate
import tiktoken
from mistralai import Mistral
import anthropic
from llamaapi import LlamaAPI
from openai import OpenAI
import google.generativeai as genai
import asyncio
import textwrap
import logging
from aiolimiter import AsyncLimiter

# Load environment variables
load_dotenv()

# Retrieve API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
llama = LlamaAPI(LLAMA_API_KEY)
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# Global rate limiter for Mistral Large (1 request every 2 seconds)
# Based on Mistral AI documentation, rate limiter should be  1 rps, however still Mistral did not provide all responses to queries.
# As a result, I increased the rate to 1 request per 2 sec.
# It significantly increased the response time of Mistral Large, however all queries got a proper response from this LLM.
mistral_rate_limiter = AsyncLimiter(max_rate=1, time_period=2)
# I changed rate limiter to 1 rps and then 1 request every 1.5 sec, but still Mistral Large didn't respond all time. 1 req / 2 sec worked.

# Pricing per 1M tokens (USD)
PRICING = {
    "OpenAI GPT-4o": {"input": 2.50, "output": 10.00},
    "Claude 3.7 Sonnet": {"input": 3.00, "output": 15.00},
    "Gemini 2.0 Flash": {"input": 0.10, "output": 0.40},
    "DeepSeek V3": {"input": 0.27, "output": 1.10},
    "LLaMA 3.3 70B": {"input": 2.80, "output": 2.80},
    "Mistral Large": {"input": 2.00, "output": 6.00},
    "Grok 2": {"input": 2.00, "output": 10.00}
}

# Tokenizer for OpenAI models
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

# Validate sentiment output
def validate_sentiment(output):
    valid_sentiments = {"positive", "negative", "neutral"}
    output = output.lower().strip(".,!?")
    for sentiment in valid_sentiments:
        if sentiment in output:
            return sentiment
    return "Error Validation"

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
            # Apply rate limiter before starting the timer
            async with mistral_rate_limiter:
                start_time = time.time()  # Start timing after rate limiter
                response = mistral_client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                end_time = time.time()  # End timing for the API call
                output = response.choices[0].message.content.strip().lower()
                output_tokens = len(openai_tokenizer.encode(output))
                api_response_time = end_time - start_time  # Calculate API response time
        else:
            # Handle other models (no rate limiter)
            start_time = time.time()  # Start timing for the API call
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

            end_time = time.time()  # End timing for the API call
            api_response_time = end_time - start_time  # Calculate API response time

        output = validate_sentiment(output)

    except Exception as e:
        return str(e), 0, 0

    cost = calculate_cost(model_name, input_tokens, output_tokens)

    # Use API response time for performance metrics (excludes rate limiter delay)
    return output, api_response_time, cost

# Function to generate summary with time and cost
async def generate_summary_with_time_cost(text, model_name, max_tokens=50, temperature=0):
    try:
        prompt = f"Provide a concise summary of the following review in one sentence, short, to the point, including key points: {text}"
        input_tokens = count_tokens(prompt)

        if model_name == "Mistral Large":
            # Apply rate limiter before starting the timer
            async with mistral_rate_limiter:
                api_start_time = time.time()  # Start timing after rate limiter
                response = mistral_client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                api_end_time = time.time()  # End timing for the API call
                output = response.choices[0].message.content.strip().lower()
                output_tokens = len(openai_tokenizer.encode(output))
                api_response_time = api_end_time - api_start_time  # Calculate API response time
        else:
            # Handle other models (no rate limiter)
            api_start_time = time.time()  # Start timing for the API call
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

            api_end_time = time.time()  # End timing for the API call
            api_response_time = api_end_time - api_start_time  # Calculate API response time

    except Exception as e:
        return str(e), 0, 0

    cost = calculate_cost(model_name, input_tokens, output_tokens)

    # Use API response time for performance metrics (excludes rate limiter delay)
    return output, api_response_time, cost

# Function to generate response with time and cost
async def generate_response_with_time_cost(review, user_name, model_name, temperature=0, max_tokens=150):
    try:
        prompt = f"You are a customer support agent. Write a short, empathetic, and informative response to the following customer review. The review mentions specific issues that the customer is experiencing. Address the customer's concerns in a clear, friendly, and professional manner, and provide suggestions or solutions where necessary. Address the customer by their name ({user_name}):\n\n{review}"
        input_tokens = count_tokens(prompt)

        if model_name == "Mistral Large":
            # Apply rate limiter before starting the timer
            async with mistral_rate_limiter:
                api_start_time = time.time()  # Start timing after rate limiter
                response = mistral_client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                api_end_time = time.time()  # End timing for the API call
                output = response.choices[0].message.content.strip()
                output_tokens = len(openai_tokenizer.encode(output))
                api_response_time = api_end_time - api_start_time  # Calculate API response time
        else:
            # Handle other models (no rate limiter)
            api_start_time = time.time()  # Start timing for the API call
            if model_name == "OpenAI GPT-4o":
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content.strip()
                output_tokens = response.usage.completion_tokens
            elif model_name == "Claude 3.7 Sonnet":
                response = anthropic_client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                output = response.content[0].text.strip()
                output_tokens = len(openai_tokenizer.encode(response.content[0].text))
            elif model_name == "Gemini 2.0 Flash":
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
                output = response.text.strip().lower()
                output_tokens = len(openai_tokenizer.encode(output))
            elif model_name == "DeepSeek V3":
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": review}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content.strip()
                output_tokens = response.usage.completion_tokens
            elif model_name == "LLaMA 3.3 70B":
                api_request_json = {
                    "model": "llama3.3-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                response = llama.run(api_request_json)
                output = response.json()["choices"][0]["message"]["content"].strip()
                output_tokens = len(openai_tokenizer.encode(output))
            elif model_name == "Grok 2":
                response = grok_client.chat.completions.create(
                    model="grok-2-latest",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": review}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content.strip()
                output_tokens = response.usage.completion_tokens
            else:
                return "Unsupported model", 0, 0

            api_end_time = time.time()  # End timing for the API call
            api_response_time = api_end_time - api_start_time  # Calculate API response time

    except Exception as e:
        return str(e), 0, 0

    cost = calculate_cost(model_name, input_tokens, output_tokens)

    # Use API response time for performance metrics (excludes rate limiter delay)
    return output, api_response_time, cost

# Function to wrap text in a specific column
def wrap_text(df, column_name, width=50):
    df[column_name] = df[column_name].apply(lambda x: "\n".join(textwrap.wrap(str(x), width)))
    return df

# Main function to calculate total time and cost for each LLM
async def calculate_total_time_and_cost(final_sample, unreplied_reviews):
    models = {
        "OpenAI GPT-4o": analyze_sentiment_with_time_cost,
        "Claude 3.7 Sonnet": analyze_sentiment_with_time_cost,
        "Gemini 2.0 Flash": analyze_sentiment_with_time_cost,
        "DeepSeek V3": analyze_sentiment_with_time_cost,
        "LLaMA 3.3 70B": analyze_sentiment_with_time_cost,
        "Mistral Large": analyze_sentiment_with_time_cost,
        "Grok 2": analyze_sentiment_with_time_cost
    }

    # Table 1: Review, LLMs, Sentiment, Summary, Generated Responses
    table1_data = []

    # Table 2: LLMs, Total Time and Cost for Each Task
    table2_data = {model: {
        "total_time_sentiment": 0,
        "total_cost_sentiment": 0,
        "total_time_summary": 0,
        "total_cost_summary": 0,
        "total_time_response": 0,
        "total_cost_response": 0,
        "total_time_end_to_end": 0,
        "total_cost_end_to_end": 0
    } for model in models.keys()}

    # Sentiment Analysis
    sentiment_results = {}  # Store sentiment results for each review

    # Create tasks for sentiment analysis
    sentiment_tasks = []
    for model_name, func in models.items():
        for text in final_sample["review"]:
            sentiment_tasks.append(func(text, model_name))

    # Run all sentiment tasks concurrently
    sentiment_results_list = await asyncio.gather(*sentiment_tasks)

    # Process sentiment results
    index = 0
    for model_name in models.keys():
        for text in final_sample["review"]:
            sentiment, response_time, response_cost = sentiment_results_list[index]
            table2_data[model_name]["total_time_sentiment"] += response_time
            table2_data[model_name]["total_cost_sentiment"] += response_cost

            if text not in sentiment_results:
                sentiment_results[text] = {}
            sentiment_results[text][model_name] = sentiment
            index += 1

    # Summarization
    summary_tasks = []
    for model_name in models.keys():
        for text in final_sample["review"]:
            summary_tasks.append(generate_summary_with_time_cost(text, model_name))

    # Run all summary tasks concurrently
    summary_results_list = await asyncio.gather(*summary_tasks)

    # Process summary results
    index = 0
    for model_name in models.keys():
        for text in final_sample["review"]:
            summary, response_time, response_cost = summary_results_list[index]
            table2_data[model_name]["total_time_summary"] += response_time
            table2_data[model_name]["total_cost_summary"] += response_cost
            index += 1

    # Response Generation
    response_tasks = []
    for model_name in models.keys():
        for index, row in unreplied_reviews.iterrows():
            review = row["review"]
            user_name = row["userName"]
            response_tasks.append(generate_response_with_time_cost(review, user_name, model_name))

    # Run all response tasks concurrently
    response_results_list = await asyncio.gather(*response_tasks)

    # Process response results
    index = 0
    for model_name in models.keys():
        for _, row in unreplied_reviews.iterrows():
            review = row["review"]
            response, response_time, response_cost = response_results_list[index]
            table2_data[model_name]["total_time_response"] += response_time
            table2_data[model_name]["total_cost_response"] += response_cost

            sentiment = sentiment_results[review][model_name]
            table1_data.append({
                "Review": review,
                "LLM": model_name,
                "Sentiment": sentiment,
                "Summary": summary_results_list[index][0],
                "Generated Response": response
            })
            index += 1

    # Calculate end-to-end time and cost for each LLM
    for model_name in models.keys():
        table2_data[model_name]["total_time_end_to_end"] = (
            table2_data[model_name]["total_time_sentiment"] +
            table2_data[model_name]["total_time_summary"] +
            table2_data[model_name]["total_time_response"]
        )
        table2_data[model_name]["total_cost_end_to_end"] = (
            table2_data[model_name]["total_cost_sentiment"] +
            table2_data[model_name]["total_cost_summary"] +
            table2_data[model_name]["total_cost_response"]
        )

    # Debugging: Print sentiment results
    for review, model_sentiments in sentiment_results.items():
        print(f"Review: {review}")
        for model_name, sentiment in model_sentiments.items():
            print(f"  {model_name}: {sentiment}")

    # Print Table 1 in the desired format
    print("\nTable 1: Review, LLMs, Sentiment, Summary, and Generated Responses")
    for review in final_sample["review"].unique():
        wrapped_review = textwrap.fill(review, width=220)  # Wrap review text
        print(f"\nReview:\n{wrapped_review}")
        review_data = [row for row in table1_data if row["Review"] == review]
        review_df = pd.DataFrame(review_data)

        # Wrap text in the "Generated Response" column to prevent excessive width
        review_df = wrap_text(review_df, "Summary", width=40)
        review_df = wrap_text(review_df, "Generated Response", width=140)

        # Print formatted table
        print(tabulate(review_df[["LLM", "Sentiment", "Summary", "Generated Response"]], 
                    headers="keys", 
                    tablefmt="grid", 
                    showindex=False, 
                    stralign="left"))

    # Print Table 2
    print("\nTable 2: LLMs, Total Time and Cost for Each Task")
    table2_df = pd.DataFrame(table2_data).T.reset_index().rename(columns={"index": "LLM"})
    print(tabulate(table2_df, 
                   headers="keys", 
                   tablefmt="grid", 
                   showindex=False, 
                   floatfmt=".4f",
                   stralign="left"))

# Filter unreplied reviews
unreplied_reviews = final_sample[final_sample["is_replied"] == "No"]

# Run the main function
asyncio.run(calculate_total_time_and_cost(final_sample, unreplied_reviews))
