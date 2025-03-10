"""
Generate Responses to Unanswered Customer Reviews with 7 Leading LLMs

This script uses seven leading LLMs to generate responses for customer reviews that have not yet been addressed.

Features:
- Identifies unanswered customer reviews.
- Generates AI-powered responses using multiple LLMs.
- Compares response quality across models.

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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

# Reference responses (multiple per user)
reference_responses = {
    "GEM Jr": [
        "Dear GEM Jr, thank you for bringing the share feature issue to our attention. We apologize for the inconvenience caused by the feature randomly selecting recently viewed items instead of your intended selections. Our team is actively working on a fix to restore the previous functionality. In the meantime, you can try manually selecting items to share. We appreciate your patience and understanding.",
        "Dear GEM Jr, we apologize for the issue with the share feature. Our team is working on a fix to ensure it works as expected. Thank you for your patience.",
        "Dear GEM Jr, we understand how frustrating it can be when the share feature doesn’t work as intended. We’re currently addressing this issue and will keep you updated on our progress. Thank you for your understanding!"
    ],
    "Nebulous_9": [
        "Dear Nebulous_9, thank you for your feedback regarding the recent changes to the delete button placement. We understand that hiding the button in a menu has made the process more inconvenient for you. This change was intended to reduce accidental deletions, but we acknowledge it may not have achieved the desired effect. We are reviewing this feedback and will consider reverting the change or improving the design. Thank you for your patience.",
        "Dear Nebulous_9, we apologize for the inconvenience caused by the recent changes to the delete button. We are actively working on improving the user experience and appreciate your feedback.",
        "Dear Nebulous_9, we hear your concerns about the delete button placement. Our team is exploring ways to make it more accessible while maintaining safety. Thank you for your patience as we work on this!"
    ],
    "Robinelmo": [
        "Dear Robinelmo, we sincerely apologize for the delay in tracking updates and delivery of your packages. We understand how frustrating it is to not know the exact location or status of your items. Our team is working to improve the tracking system to provide real-time updates. In the meantime, please feel free to contact our support team for assistance with your current orders. Thank you for your patience.",
        "Dear Robinelmo, we apologize for the delay in tracking and delivery. Our team is working hard to resolve this issue and improve the system. Thank you for your understanding.",
        "Dear Robinelmo, we regret the inconvenience caused by the delay in tracking updates. Our team is implementing new measures to ensure real-time updates in the future. Thank you for your patience!"
    ],
    "TYPERMX": [
        "Dear TYPERMX, thank you for your suggestion about adding sound notifications for delivery updates. We agree that audible alerts for updates like 'drop off on its way' or '10 stops delivered' would enhance the user experience. We will share your feedback with our development team and consider implementing this feature in a future update. We appreciate your input!",
        "Dear TYPERMX, we appreciate your suggestion for sound notifications. Our team will review this feature request and consider it for future updates. Thank you for your feedback!",
        "Dear TYPERMX, your idea for sound notifications is fantastic! We’re exploring ways to integrate this feature into our app. Thank you for sharing your thoughts!"
    ],
    "Labratlp": [
        "Dear Labratlp, we apologize for the inconvenience caused by the issue with the menu and account buttons after the latest update. Our technical team is aware of the problem and is working on a fix. In the meantime, you can try clearing the app cache or reinstalling the app to see if it resolves the issue. Thank you for your patience and understanding.",
        "Dear Labratlp, we apologize for the issue with the menu and account buttons. Our team is working on a fix, and we appreciate your patience while we resolve this.",
        "Dear Labratlp, we understand how frustrating it can be when the menu and account buttons don’t work as expected. Our team is actively addressing this issue. Thank you for your patience!"
    ],
    "Michelle Marcotte": [
        "Dear Michelle Marcotte, thank you for your feedback regarding your delivery experience during the Polar Vortex. We are glad to hear that despite the extreme weather conditions, your deliveries were only delayed by a day. Our delivery teams work hard to ensure timely service, even in challenging conditions. We appreciate your understanding and support.",
        "Dear Michelle Marcotte, we are pleased to hear that your deliveries were only slightly delayed during the Polar Vortex. Thank you for your patience and support!",
        "Dear Michelle Marcotte, we’re grateful for your understanding during the Polar Vortex. Our team is committed to ensuring timely deliveries, no matter the weather. Thank you for your support!"
    ],
    "K": [
        "مرحبًا K، نعتذر بشدة عن التجربة التي واجهتها مع خدمة الشحن وخدمة العملاء. نتفهم مدى أهمية سرعة الشحن بالنسبة لك، وسنعمل على تحسين تجربتك معنا. سنقوم بمراجعة شركة النقل المعنية وضمان تسليم أسرع في المستقبل. إذا كنت بحاجة إلى أي مساعدة إضافية، فلا تتردد في التواصل معنا مباشرة. نشكرك على تعليقاتك القيمة، فهي تساعدنا على التحسين.",
        "عزيزي K، نشكرك على مشاركة تجربتك معنا. نأسف لسماع أن خدمة الشحن لم تكن كما توقعتها. لمساعدتك، سنقوم بإعادة شحن طلبك مع ناقل آخر لضمان وصوله بشكل أسرع. بالإضافة إلى ذلك، سننقل ملاحظاتك حول خدمة العملاء إلى الفريق المعني لتحسين الخدمة. نأمل أن نتمكن من استعادة ثقتك بنا. إذا كان لديك أي استفسارات أخرى، فنحن هنا لمساعدتك.",
        "K العزيز، نشكرك على تعليقاتك الصادقة. نتفهم إحباطك بشأن سرعة الشحن وخدمة العملاء. نعمل حاليًا على تحسين تجربة الشحن لدينا وضمان وصول الطلبات بشكل أسرع في المستقبل. كما سنقوم بتدريب فريق خدمة العملاء لتقديم خدمة أفضل. نأمل أن نتمكن من تقديم تجربة أكثر إرضاءً في المرات القادمة. شكرًا لصبرك وتفهمك."
    ],
    "Ivory James Jones IV": [
        "Dear Ivory James Jones IV, thank you for your positive feedback about our shopping, shipping, and delivery services. We are thrilled to hear that you are satisfied with our services and appreciate your support. If there’s anything more we can do to enhance your experience, please let us know. Thank you for choosing us!",
        "Dear Ivory James Jones IV, we are delighted to hear that you are happy with our services. Thank you for your positive feedback and support!",
        "Dear Ivory James Jones IV, your kind words mean a lot to us! We’re committed to providing the best shopping experience. Thank you for your support!"
    ],
    "Molly Garrett": [
        "Dear Molly Garrett, we apologize for the frustration caused by the issue with uninstalling the app. To resolve this, please ensure that the app does not have administrator privileges enabled. You can check this in your device settings under 'Apps' or 'Security.' If you need further assistance, please contact our support team. Thank you for your patience.",
        "Dear Molly Garrett, we apologize for the uninstall issue. Please check your device settings for administrator privileges or contact our support team for help. Thank you for your patience.",
        "Dear Molly Garrett, we’re sorry for the trouble you’re having with uninstalling the app. Our team is working on a solution. Thank you for your patience!"
    ],
    "MD SAYEDUL AFKAHIN": [
        "Dear MD SAYEDUL AFKAHIN, thank you for your feedback regarding the need for our services in Bangladesh. We are continuously working to expand our global presence, and your input is valuable to us. While we do not currently have a timeline for expansion into Bangladesh, we will certainly consider your request as we plan future updates. Thank you for your support!",
        "Dear MD SAYEDUL AFKAHIN, we appreciate your feedback about expanding our services to Bangladesh. While we don’t have a timeline yet, we will consider your request in our future plans. Thank you for your support!",
        "Dear MD SAYEDUL AFKAHIN, we’re excited about the possibility of expanding to Bangladesh! Your feedback is invaluable as we plan our next steps. Thank you for your support!"
    ]
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

# Function to compute sentiment
def compute_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to compute BLEU Score
def compute_bleu(reference, generated):
    reference_tokens = reference.split()
    generated_tokens = generated.split()
    return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=SmoothingFunction().method1)

# Function to select the best reference response
def select_best_reference(generated_response, reference_responses):
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    generated_embedding = similarity_model.encode([generated_response])
    reference_embeddings = similarity_model.encode(reference_responses)
    similarities = cosine_similarity(generated_embedding, reference_embeddings)
    best_index = similarities.argmax()
    return reference_responses[best_index]

# Function to calculate sentiment alignment
def calculate_sentiment_alignment(review_sentiment, response_sentiment):
    return 1 if review_sentiment == response_sentiment else 0

# Function to generate response with time and cost
async def generate_response_with_time_cost(review, user_name, model_name, max_tokens=150, temperature=0):
    try:
        prompt = (
            f"You are a customer support agent. Write a short, empathetic, and informative response to the following customer review. "
            f"The review mentions specific issues that the customer is experiencing. Address the customer's concerns in a clear, friendly, and professional manner, "
            f"and provide suggestions or solutions where necessary. Address the customer by their name ({user_name}):\n\n"
            f"Review: {review}\n\n"
            f"Response:"
        )
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
                    messages=[{"role": "user", "content": prompt}],
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
                output = response.text.strip()
                output_tokens = len(openai_tokenizer.encode(output))
            elif model_name == "DeepSeek V3":
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content.strip()
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
                output = response.json()["choices"][0]["message"]["content"].strip()
                output_tokens = len(openai_tokenizer.encode(output))
            elif model_name == "Grok 2":
                response = grok_client.chat.completions.create(
                    model="grok-2-latest",
                    messages=[{"role": "user", "content": prompt}],
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

# Main function to calculate total time and cost for responses
async def calculate_response_time_and_cost(final_sample):
    models = [
        "OpenAI GPT-4o",
        "Claude 3.7 Sonnet",
        "Gemini 2.0 Flash",
        "LLaMA 3.3 70B",
        "Mistral Large",
        "DeepSeek V3",
        "Grok 2"
    ]

    # Table 1: Review, LLMs, and Responses
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

    # Focus on unreplied reviews
    unreplied_reviews = final_sample[final_sample["is_replied"] == "No"]

    # Response generation
    response_tasks = []
    for model_name in models:
        for _, row in unreplied_reviews.iterrows():
            review = row["review"]
            user_name = row["userName"]
            response_tasks.append(generate_response_with_time_cost(review, user_name, model_name))

    # Run all response tasks concurrently
    response_results_list = await asyncio.gather(*response_tasks)

    # Process response results
    index = 0
    for model_name in models:
        for _, row in unreplied_reviews.iterrows():
            review = row["review"]
            user_name = row["userName"]
            response, response_time, response_cost = response_results_list[index]

            # Calculate metrics
            reference = select_best_reference(response, reference_responses.get(user_name, ["No reference available"]))
            rouge_scores = compute_rouge(reference, response)
            bertscore = compute_bertscore(reference, response)
            meteor = compute_meteor(reference, response)
            perplexity = calculate_perplexity(response)
            compression_ratio = compute_compression_ratio(review, response)

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
            if not any(row["Review"] == review for row in table1_data):
                table1_data.append({
                    "Review": review,
                    "OpenAI GPT-4o": "",
                    "Claude 3.7 Sonnet": "",
                    "Gemini 2.0 Flash": "",
                    "LLaMA 3.3 70B": "",
                    "Mistral Large": "",
                    "DeepSeek V3": "",
                    "Grok 2": ""
                })
            # Find the row for this review and update the corresponding model's response
            for row in table1_data:
                if row["Review"] == review:
                    row[model_name] = response
                    break
            index += 1

    # Print Table 1 in the desired format
    print("\nTable 1: Review and Responses by LLM")
    table1_df = pd.DataFrame(table1_data)
    table1_df = wrap_text(table1_df, "Review", width=40)  # Wrap review text
    table1_df = wrap_text(table1_df, "OpenAI GPT-4o", width=40)  # Wrap response text
    table1_df = wrap_text(table1_df, "Claude 3.7 Sonnet", width=40)  # Wrap response text
    table1_df = wrap_text(table1_df, "Gemini 2.0 Flash", width=40)  # Wrap response text
    table1_df = wrap_text(table1_df, "LLaMA 3.3 70B", width=40)  # Wrap response text
    table1_df = wrap_text(table1_df, "Mistral Large", width=40)  # Wrap response text
    table1_df = wrap_text(table1_df, "DeepSeek V3", width=40)  # Wrap response text
    table1_df = wrap_text(table1_df, "Grok 2", width=40)  # Wrap response text
    print(tabulate(table1_df, headers="keys", tablefmt="grid", showindex=False, stralign="left"))

    # Print Table 2
    print("\nTable 2: Performance Analytics by LLM")
    table2_df = pd.DataFrame(table2_data).T.reset_index().rename(columns={"index": "LLM"})
    print(tabulate(table2_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f", stralign="left"))

# Run the main function
asyncio.run(calculate_response_time_and_cost(final_sample))
