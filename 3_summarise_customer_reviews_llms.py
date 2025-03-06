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
from mistralai import Mistral
from dotenv import load_dotenv
import anthropic
import json
from llamaapi import LlamaAPI
from openai import OpenAI
import google.generativeai as genai
from tabulate import tabulate
from textwrap import fill

# For performance abalytics
from rouge import Rouge
from bert_score import score

# For response time and cost analytics
from nltk.translate.meteor_score import meteor_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import tiktoken

# ---------------------------------------------------------------------
# Section 0- Configurations and setups
# ---------------------------------------------------------------------

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

# Initialise API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
llama = LlamaAPI(LLAMA_API_KEY)
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# ---------------------------------------------------------------------
# Section 1- Summarise customer reviews by each of leading LLMs
# ---------------------------------------------------------------------

# Standardised Summary Prompt
SUMMARY_PROMPT = "Provide a concise summary of the following review in one sentence, short, to the point, including key points:"

# Function to generate a summary of the review
def generate_summary(text, model_name, max_tokens=30, temperature=0):
    """
    Generate a summary of the input text using the specified LLM.
    
    Args:
        text (str): The input text to summarise.
        model_name (str): The name of the LLM to use.
        max_tokens (int): Maximum number of tokens for the output summary.
        temperature (float): Controls randomness in the output (0 for deterministic output).
    
    Returns:
        str: The generated summary or an error message.
    """
    try:
        # Standardised prompt for all models
        prompt = f"{SUMMARY_PROMPT} {text}"
        
        if model_name == "OpenAI GPT-4o":
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SUMMARY_PROMPT},
                          {"role": "user", "content": text}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        
        elif model_name == "Claude 3.7 Sonnet":
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.content[0].text.strip()
        
        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
            )
            return response.text.strip()
        
        elif model_name == "DeepSeek V3":
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        
        elif model_name == "LLaMA 3.3 70B":
            api_request_json = {
                "model": "llama3.3-70b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            response = llama.run(api_request_json)
            response_json = response.json()
            if "choices" in response_json and response_json["choices"]:
                return response_json["choices"][0]["message"]["content"].strip()
            return f"Error: Unexpected response format - {json.dumps(response_json, indent=2)}"
        
        elif model_name == "Mistral Large":
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return chat_response.choices[0].message.content.strip()
        
        elif model_name == "Grok 2":
            response = grok_client.chat.completions.create(
                model="grok-2-latest",
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        
        else:
            return "Unsupported model"
    except Exception as e:
        return str(e)

# Define the models to evaluate
models = {
    "OpenAI GPT-4o": generate_summary,
    "Claude 3.7 Sonnet": generate_summary,
    "Gemini 2.0 Flash": generate_summary,
    "LLaMA 3.3 70B": generate_summary,
    "Mistral Large": generate_summary,
    "DeepSeek V3": generate_summary,
    "Grok 2": generate_summary
}

# Store results
results = []

# Iterate through each review and model
for index, row in final_sample.iterrows():
    review = row["review"]
    for model_name, func in models.items():
        summary = func(review, model_name)
        results.append({
            "Review": review,
            "LLM": model_name,
            "Summary": summary
        })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Wrap long text for better readability
results_df["Review"] = results_df["Review"].apply(lambda x: fill(x, width=80))
results_df["Summary"] = results_df["Summary"].apply(lambda x: fill(x, width=80))

# Print the results in a clear table format
print("\nReview Summarization Results:")
print(tabulate(
    results_df[["Review", "LLM", "Summary"]],
    headers="keys",
    tablefmt="grid",
    showindex=False,
    colalign=("left", "left", "left")  # Left-align all columns
))

# ---------------------------------------------------------------------
# Section 2- Performance Analysis- "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore", "Compression Ratio"
# ---------------------------------------------------------------------

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
    P, R, F1 = score([generated], [reference], lang="en")
    return F1.mean().item()

# Function to compute Compression Ratio
def compute_compression_ratio(original_text, summary):
    return len(original_text.split()) / len(summary.split())

# Store generated summary results
results = []

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

# Iterate through each review and model
for index, row in final_sample.iterrows():
    review = row["review"]
    reference_summary = row["reference_summary"]  # Ensure this column exists
    for model_name, func in models.items():
        generated_summary = func(review, model_name)
        
        # Compute metrics
        rouge_scores = compute_rouge(reference_summary, generated_summary)
        bertscore = compute_bertscore(reference_summary, generated_summary)
        compression_ratio = compute_compression_ratio(review, generated_summary)
        
        # Add human evaluation placeholders (to be filled manually)
        coherence = None  # Replace with human rating (1-5)
        relevance = None  # Replace with human rating (1-5)
        
        results.append({
            "Review": review,
            "LLM": model_name,
            "Generated Summary": generated_summary,
            "ROUGE-1 F1": rouge_scores["rouge-1"],
            "ROUGE-2 F1": rouge_scores["rouge-2"],
            "ROUGE-L F1": rouge_scores["rouge-l"],
            "BERTScore": bertscore,
            "Compression Ratio": compression_ratio
            # ,"Coherence (Human)": coherence
            # ,"Relevance (Human)": relevance
        })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Wrap long text for better readability
results_df["Review"] = results_df["Review"].apply(lambda x: fill(x, width=80))
results_df["Generated Summary"] = results_df["Generated Summary"].apply(lambda x: fill(x, width=80))

# Print the results in a clear table format
print("\nPerformance Comparison of LLMs for Summarization:")
print(tabulate(
    # results_df[["LLM", "Review", "Generated Summary", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore", "Compression Ratio", "Coherence (Human)", "Relevance (Human)"]],
        results_df[["LLM", "Review", "Generated Summary", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore", "Compression Ratio"]],
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))

# After generating results for all reviews, calculate overall averages for each LLM
overall_scores = {}

# Group results by LLM
for model_name in models.keys():
    model_results = results_df[results_df["LLM"] == model_name]
    
    # Calculate averages for each metric
    overall_scores[model_name] = {
        "ROUGE-1 F1": model_results["ROUGE-1 F1"].mean(),
        "ROUGE-2 F1": model_results["ROUGE-2 F1"].mean(),
        "ROUGE-L F1": model_results["ROUGE-L F1"].mean(),
        "BERTScore": model_results["BERTScore"].mean(),
        "Compression Ratio": model_results["Compression Ratio"].mean()
    }

# Convert the overall scores to a DataFrame for better visualization
overall_scores_df = pd.DataFrame(overall_scores).T.reset_index()
overall_scores_df.rename(columns={"index": "LLM"}, inplace=True)

# Print the overall scores in a clear table format
print("\nOverall Performance Comparison of LLMs for Summarization:")
print(tabulate(
    overall_scores_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))

# ---------------------------------------------------------------------
# Section 3- Add time and cost analytics to above performance metrics
# ---------------------------------------------------------------------

# Load GPT-2 model for perplexity calculation
# GPT-2 for perplexity calculation is a common practice in NLP
# It’s lightweight, free, and provides consistent results. 
# If you need higher accuracy or are working on a high-stakes project, you can consider using GPT-4 or GPT-4o, keeping in mind the additional costs and complexity
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to calculate perplexity
def calculate_perplexity(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return float(torch.exp(loss))  # Perplexity = exp(loss)

# Function to compute METEOR score
def compute_meteor(reference, generated):
    return meteor_score([reference.split()], generated.split())

# Function to calculate cost based on input and output tokens
def calculate_cost(model_name, input_tokens, output_tokens):
    input_price = PRICING[model_name]["input"] / 1_000_000  # Convert to cost per token
    output_price = PRICING[model_name]["output"] / 1_000_000  # Convert to cost per token
    total_cost = (input_tokens * input_price) + (output_tokens * output_price)
    return round(total_cost, 6)

# Pricing per 1M tokens (USD)- Pricing references are availabe in Resources section of the article
PRICING = {
    "OpenAI GPT-4o": {"input": 2.50, "output": 10.00},  # Input: $2.50, Output: $10.00
    "Claude 3.7 Sonnet": {"input": 3.00, "output": 15.00},  # Input: $3.00, Output: $15.00
    "Gemini 2.0 Flash": {"input": 0.10, "output": 0.40},  # Input: $0.10, Output: $0.40
    "DeepSeek V3": {"input": 0.27, "output": 1.10},  # Cache Miss: $0.27, Output: $1.10
    "LLaMA 3.3 70B": {"input": 2.80, "output": 2.80},  # Input: $2.80, Output: $2.80
    "Mistral Large": {"input": 2.00, "output": 6.00},  # Input: $2.00, Output: $6.00
    "Grok 2": {"input": 2.00, "output": 10.00}  # Input: $2.00, Output: $10.00
}

# Models to evaluate
MODELS = list(PRICING.keys())

# Tokenizer for OpenAI models
openai_tokenizer = tiktoken.encoding_for_model("gpt-4")

# Function to calculate cost based on input and output tokens
def calculate_cost(model_name, input_tokens, output_tokens):
    input_price = PRICING[model_name]["input"] / 1_000_000  # Convert to cost per token
    output_price = PRICING[model_name]["output"] / 1_000_000  # Convert to cost per token
    total_cost = (input_tokens * input_price) + (output_tokens * output_price)
    return round(total_cost, 6)

# Function to count tokens for OpenAI models
def count_tokens(text):
    return len(openai_tokenizer.encode(text))

# Function to call the model and measure time and cost
def call_model(text, model_name, task, max_tokens=50, temperature=0):
    start_time = time.time()
    try:
        prompt = SUMMARY_PROMPT + " " + text
        input_tokens = count_tokens(prompt)  # Accurate token counting for OpenAI models

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
            output_tokens = len(response.content[0].text.split())  # Approximate output tokens

        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
            )
            output = response.text.strip().lower()
            output_tokens = len(output.split())  # Approximate output tokens

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
            output_tokens = len(output.split())  # Approximate output tokens

        elif model_name == "Mistral Large":
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            output = chat_response.choices[0].message.content.strip().lower()
            output_tokens = len(output.split())  # Approximate output tokens

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
            return model_name, "Unsupported model", 0, 0

    except Exception as e:
        return model_name, str(e), 0, 0

    end_time = round(time.time() - start_time, 4)
    cost = calculate_cost(model_name, input_tokens, output_tokens)

    return model_name, output, end_time, cost

# Main loop with all metrics
results = []

for index, row in final_sample.iterrows():
    review = row["review"]
    reference_summary = row["reference_summary"]  # Ensure this column exists
    for model_name, func in models.items():
        # Generate summary and measure time/cost
        model_result = call_model(review, model_name, "summary")
        generated_summary = model_result[1]
        response_time = model_result[2]
        response_cost = model_result[3]
        
        # Compute metrics
        rouge_scores = compute_rouge(reference_summary, generated_summary)
        bertscore = compute_bertscore(reference_summary, generated_summary)
        meteor = compute_meteor(reference_summary, generated_summary)
        perplexity = calculate_perplexity(generated_summary)
        compression_ratio = compute_compression_ratio(review, generated_summary)
        
        # Add results
        results.append({
            "Review": review,
            "LLM": model_name,
            "Generated Summary": generated_summary,
            "ROUGE-1 F1": rouge_scores["rouge-1"],
            "ROUGE-2 F1": rouge_scores["rouge-2"],
            "ROUGE-L F1": rouge_scores["rouge-l"],
            "BERTScore": bertscore,
            "METEOR": meteor,
            "Perplexity": perplexity,
            "Compression Ratio": compression_ratio,
            "Response Time (s)": response_time,
            "Response Cost (USD)": response_cost
            # ,"Coherence (Human)": None,  # Placeholder for human evaluation
            # ,"Relevance (Human)": None   # Placeholder for human evaluation
        })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Wrap long text for better readability
results_df["Review"] = results_df["Review"].apply(lambda x: fill(x, width=80))
results_df["Generated Summary"] = results_df["Generated Summary"].apply(lambda x: fill(x, width=80))

# Print the results in a clear table format
print("\nPerformance Comparison of LLMs for Summarization:")
print(tabulate(
    # results_df[["LLM", "Review", "Generated Summary", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore", "METEOR", "Perplexity", "Compression Ratio", "Response Time (s)", "Response Cost (USD)", "Coherence (Human)", "Relevance (Human)"]],
    results_df[["LLM", "Review", "Generated Summary", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore", "METEOR", "Perplexity", "Compression Ratio", "Response Time (s)", "Response Cost (USD)"]],
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".5f"
))

# Calculate overall averages for each LLM
overall_scores = {}

# Group results by LLM
for model_name in models.keys():
    model_results = results_df[results_df["LLM"] == model_name]
    
    # Calculate averages for each metric
    overall_scores[model_name] = {
        "ROUGE-1 F1": model_results["ROUGE-1 F1"].mean(),
        "ROUGE-2 F1": model_results["ROUGE-2 F1"].mean(),
        "ROUGE-L F1": model_results["ROUGE-L F1"].mean(),
        "BERTScore": model_results["BERTScore"].mean(),
        "METEOR": model_results["METEOR"].mean(),
        "Perplexity": model_results["Perplexity"].mean(),
        "Compression Ratio": model_results["Compression Ratio"].mean(),
        "Total Response Time (s)": model_results["Response Time (s)"].sum(),  # Summing response time
        "Total Response Cost (USD)": model_results["Response Cost (USD)"].sum()  # Summing cost
    }

# Convert the overall scores to a DataFrame for better visualization
overall_scores_df = pd.DataFrame(overall_scores).T.reset_index()
overall_scores_df.rename(columns={"index": "LLM"}, inplace=True)

# Print the overall scores in a clear table format
print("\nOverall Performance Comparison of LLMs for Summarization:")
print(tabulate(
    overall_scores_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".5f"
))
