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

# Initialise API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
llama = LlamaAPI(LLAMA_API_KEY)
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

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

# Tokeniser for OpenAI models
openai_tokeniser = tiktoken.encoding_for_model("gpt-4")

# Function to count tokens for OpenAI models
def count_tokens(text):
    return len(openai_tokeniser.encode(text))

# Function to calculate cost based on input and output tokens
def calculate_cost(model_name, input_tokens, output_tokens):
    input_price = PRICING[model_name]["input"] / 1_000_000  # Convert to cost per token
    output_price = PRICING[model_name]["output"] / 1_000_000  # Convert to cost per token
    total_cost = (input_tokens * input_price) + (output_tokens * output_price)
    return round(total_cost, 6)

# Function to analyze sentiment with time and cost
async def analyze_sentiment_with_time_cost(text, model_name, max_tokens=10, temperature=0):
    start_time = time.time()
    try:
        prompt = f"Analyze the sentiment of the following text and return only 'positive', 'negative', or 'neutral': {text}"
        input_tokens = count_tokens(prompt)

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
                messages=[{"role": "user", "content": prompt}]
            )
            output = response.content[0].text.strip().lower()
            output_tokens = len(openai_tokeniser.encode(response.content[0].text))

        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
            output = response.text.strip().lower()
            output_tokens = len(openai_tokeniser.encode(output))

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
            api_request_json = {"model": "llama3.3-70b", "messages": [{"role": "user", "content": prompt}], "stream": False}
            response = llama.run(api_request_json)
            output = response.json()["choices"][0]["message"]["content"].strip().lower()
            output_tokens = len(openai_tokeniser.encode(output))

        elif model_name == "Mistral Large":
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            output = chat_response.choices[0].message.content.strip().lower()
            output_tokens = len(openai_tokeniser.encode(output))

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

    except Exception as e:
        return str(e), 0, 0

    end_time = round(time.time() - start_time, 4)
    cost = calculate_cost(model_name, input_tokens, output_tokens)

    return output, end_time, cost

# Function to generate summary with time and cost
async def generate_summary_with_time_cost(text, model_name, max_tokens=50, temperature=0):
    start_time = time.time()
    try:
        prompt = f"Provide a concise summary of the following review in one sentence, short, to the point, including key points: {text}"
        input_tokens = count_tokens(prompt)

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
            output_tokens = len(openai_tokeniser.encode(response.content[0].text))

        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
            output = response.text.strip().lower()
            output_tokens = len(openai_tokeniser.encode(output))

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
            api_request_json = {"model": "llama3.3-70b", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "stream": False}
            response = llama.run(api_request_json)
            output = response.json()["choices"][0]["message"]["content"].strip().lower()
            output_tokens = len(openai_tokeniser.encode(output))

        elif model_name == "Mistral Large":
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            output = chat_response.choices[0].message.content.strip().lower()
            output_tokens = len(openai_tokeniser.encode(output))

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

    except Exception as e:
        return str(e), 0, 0

    end_time = round(time.time() - start_time, 4)
    cost = calculate_cost(model_name, input_tokens, output_tokens)

    return output, end_time, cost

# Function to generate response with time and cost
async def generate_response_with_time_cost(review, user_name, model_name, temperature=0, max_tokens=150):
    start_time = time.time()
    try:
        prompt = f"You are a customer support agent. Write a short, empathetic, and informative response to the following customer review. The review mentions specific issues that the customer is experiencing. Address the customer's concerns in a clear, friendly, and professional manner, and provide suggestions or solutions where necessary. Address the customer by their name ({user_name}):\n\n{review}"
        input_tokens = count_tokens(prompt)

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
            output_tokens = len(openai_tokeniser.encode(response.content[0].text))

        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
            output = response.text.strip().lower()
            output_tokens = len(openai_tokeniser.encode(output))

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
            api_request_json = {"model": "llama3.3-70b", "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens, "stream": False}
            response = llama.run(api_request_json)
            output = response.json()["choices"][0]["message"]["content"].strip()
            output_tokens = len(openai_tokeniser.encode(output))

        elif model_name == "Mistral Large":
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": review}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            output = chat_response.choices[0].message.content.strip()
            output_tokens = len(openai_tokeniser.encode(output))

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

    except Exception as e:
        return str(e), 0, 0

    end_time = round(time.time() - start_time, 4)
    cost = calculate_cost(model_name, input_tokens, output_tokens)

    return output, end_time, cost

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
    for model_name, func in models.items():
        for text in final_sample["review"]:
            sentiment, response_time, response_cost = await func(text, model_name)
            table2_data[model_name]["total_time_sentiment"] += response_time
            table2_data[model_name]["total_cost_sentiment"] += response_cost

    # Summarisation
    for model_name, func in models.items():
        for text in final_sample["review"]:
            summary, response_time, response_cost = await generate_summary_with_time_cost(text, model_name)
            table2_data[model_name]["total_time_summary"] += response_time
            table2_data[model_name]["total_cost_summary"] += response_cost

    # Response Generation
    for model_name, func in models.items():
        for index, row in unreplied_reviews.iterrows():
            review = row["review"]
            user_name = row["userName"]
            response, response_time, response_cost = await generate_response_with_time_cost(review, user_name, model_name)
            table2_data[model_name]["total_time_response"] += response_time
            table2_data[model_name]["total_cost_response"] += response_cost

            # Add data to Table 1
            table1_data.append({
                "Review": review,
                "LLM": model_name,
                "Sentiment": sentiment,
                "Summary": summary,
                "Generated Response": response
            })

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

    # Print Table 1 in the desired format
    print("\nTable 1: Review, LLMs, Sentiment, Summary, and Generated Responses")
    for review in final_sample["review"].unique():
        print(f"\nReview: {review}")
        review_data = [row for row in table1_data if row["Review"] == review]
        review_df = pd.DataFrame(review_data)
        print(tabulate(review_df[["LLM", "Sentiment", "Summary", "Generated Response"]], headers="keys", tablefmt="grid", showindex=False))

    # Print Table 2
    print("\nTable 2: LLMs, Total Time and Cost for Each Task")
    table2_df = pd.DataFrame(table2_data).T.reset_index().rename(columns={"index": "LLM"})
    print(tabulate(table2_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

# Filter unreplied reviews
unreplied_reviews = final_sample[final_sample["is_replied"] == "No"]

# Run the main function
# To run the tasks concurrently, improving performance
asyncio.run(calculate_total_time_and_cost(final_sample, unreplied_reviews))
