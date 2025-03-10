"""
Summarise Customer Reviews

This script extracts and summarises customer reviews using leading LLMs to generate concise insights from large volumes of feedback.

Features:
- Uses 7 LLMs for automated summarisation.
- Performances, response time, and cost are compared for these llms.
- Outputs structured summaries for easy analysis.

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
from aiolimiter import AsyncLimiter
from rouge import Rouge
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

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
mistral_rate_limiter = AsyncLimiter(max_rate=1, time_period=2)

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

reference_summary = [ # Created by studying reviews and providing summary for them
    "The share feature is malfunctioning, randomly selecting recently viewed items instead of allowing specific choices, making it difficult to share gift ideas via text.",
    "Hiding the delete button in a menu is a cheap tactic that adds inconvenience without increasing purchases.",
    "The tracking system is outdated, with delayed updates and packages not arriving on time.",
    "The app works well, but silent notifications for delivery updates are inconvenient.",
    "The latest update broke the menu and account buttons, making the app difficult to use for shopping.",
    "Amazon performed reliably during the Polar Vortex, with only minor delays.",
    "المستخدم يشعر بخيبة أمل بسبب عدم تحقيق برايم لتوقعاته فيما يتعلق بسرعة الشحن وجودة خدمة العملاء",
    "The app provides excellent shopping, shipping, and delivery services.",
    "The user is frustrated because they cannot uninstall the app and makes an unfounded accusation about Elon Musk.",
    "The user requests the app to be made available in Bangladesh."
]
final_sample["reference_summary"]= reference_summary

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

# Function to compute ROUGE scores
def compute_rouge(reference, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    return {
        "rouge-1": scores[0]["rouge-1"]["f"],
        "rouge-2": scores[0]["rouge-2"]["f"],
        "rouge-l": scores[0]["rouge-l"]["f"]
    }

# Function to compute BERTScore
def compute_bertscore(reference, generated):
    P, R, F1 = bert_score([generated], [reference], lang="en")
    return F1.mean().item()

# Function to compute METEOR score
def compute_meteor(reference, generated):
    return meteor_score([reference.split()], generated.split())

# Function to compute Compression Ratio
def compute_compression_ratio(original_text, summary):
    return len(original_text.split()) / len(summary.split())

# Load GPT-2 model for perplexity calculation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to calculate perplexity
def calculate_perplexity(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return float(torch.exp(loss))  # Perplexity = exp(loss)

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

# Function to wrap text in a specific column
def wrap_text(df, column_name, width=50):
    df[column_name] = df[column_name].apply(lambda x: "\n".join(textwrap.wrap(str(x), width)))
    return df

# Main function to calculate total time and cost for summaries
async def calculate_summary_time_and_cost(final_sample):
    models = [
        "OpenAI GPT-4o",
        "Claude 3.7 Sonnet",
        "Gemini 2.0 Flash",
        "LLaMA 3.3 70B",
        "Mistral Large",
        "DeepSeek V3",
        "Grok 2"
    ]

    # Table 1: Review, LLMs, and Summaries
    table1_data = []

    # Table 2: Performance analytics for each LLM
    table2_data = {model: {
        "ROUGE-1 F1": 0,
        "ROUGE-2 F1": 0,
        "ROUGE-L F1": 0,
        "BERTScore": 0,
        "METEOR": 0,
        "Perplexity": 0,
        "Compression Ratio": 0,
        "Total Response Time (s)": 0,
        "Total Response Cost (USD)": 0
    } for model in models}

    # Summarization
    summary_tasks = []
    for model_name in models:
        for text in final_sample["review"]:
            summary_tasks.append(generate_summary_with_time_cost(text, model_name))

    # Run all summary tasks concurrently
    summary_results_list = await asyncio.gather(*summary_tasks)

    # Process summary results
    index = 0
    for model_name in models:
        for text in final_sample["review"]:
            summary, response_time, response_cost = summary_results_list[index]

            # Calculate metrics
            reference = final_sample[final_sample["review"] == text]["reference_summary"].values[0]  # Assuming you have reference summaries
            rouge_scores = compute_rouge(reference, summary)
            bertscore = compute_bertscore(reference, summary)
            meteor = compute_meteor(reference, summary)
            perplexity = calculate_perplexity(summary)
            compression_ratio = compute_compression_ratio(text, summary)

            # Update Table 2 data
            table2_data[model_name]["ROUGE-1 F1"] += rouge_scores["rouge-1"]
            table2_data[model_name]["ROUGE-2 F1"] += rouge_scores["rouge-2"]
            table2_data[model_name]["ROUGE-L F1"] += rouge_scores["rouge-l"]
            table2_data[model_name]["BERTScore"] += bertscore
            table2_data[model_name]["METEOR"] += meteor
            table2_data[model_name]["Perplexity"] += perplexity
            table2_data[model_name]["Compression Ratio"] += compression_ratio
            table2_data[model_name]["Total Response Time (s)"] += response_time
            table2_data[model_name]["Total Response Cost (USD)"] += response_cost

            # Add to Table 1
            if not any(row["Review"] == text for row in table1_data):
                table1_data.append({
                    "Review": text,
                    "OpenAI GPT-4o": "",
                    "Claude 3.7 Sonnet": "",
                    "Gemini 2.0 Flash": "",
                    "LLaMA 3.3 70B": "",
                    "Mistral Large": "",
                    "DeepSeek V3": "",
                    "Grok 2": ""
                })
            # Find the row for this review and update the corresponding model's summary
            for row in table1_data:
                if row["Review"] == text:
                    row[model_name] = summary
                    break
            index += 1

    # Print Table 1 in the desired format
    print("\nTable 1: Review and Summaries by LLM")
    table1_df = pd.DataFrame(table1_data)
    table1_df = wrap_text(table1_df, "Review", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "OpenAI GPT-4o", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "Claude 3.7 Sonnet", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "Gemini 2.0 Flash", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "LLaMA 3.3 70B", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "Mistral Large", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "DeepSeek V3", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "Grok 2", width=40)  # Wrap review text
    print(tabulate(table1_df, headers="keys", tablefmt="grid", showindex=False, stralign="left"))

    # Print Table 2
    print("\nTable 2: Performance Analytics by LLM")
    table2_df = pd.DataFrame(table2_data).T.reset_index().rename(columns={"index": "LLM"})
    print(tabulate(table2_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f", stralign="left"))

# Run the main function
asyncio.run(calculate_summary_time_and_cost(final_sample))
