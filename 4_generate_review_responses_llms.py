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
from mistralai import Mistral
from dotenv import load_dotenv
import anthropic
import json
from llamaapi import LlamaAPI
from openai import OpenAI
import google.generativeai as genai
from rouge import Rouge
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel
from textblob import TextBlob
from tabulate import tabulate
import torch  # Import torch for perplexity calculation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tiktoken  # For accurate token counting

# -----------------------------------------------------------------------
# Section 0- Configurations and setups
# -----------------------------------------------------------------------

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

# -----------------------------------------------------------------------
# Section 1- Generate Responses- Reviews that haven't been responded yet
# -----------------------------------------------------------------------

# Function to generate a response to a customer review
def generate_response(review, user_name, model_name, temperature=0, max_tokens=150):
    try:
        prompt = f"You are a customer support agent. Write a short, empathetic, and informative response to the following customer review. The review mentions specific issues that the customer is experiencing. Address the customer's concerns in a clear, friendly, and professional manner, and provide suggestions or solutions where necessary. Address the customer by their name ({user_name}):\n\n{review}"
        
        if model_name == "OpenAI GPT-4o":
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=max_tokens,  # Added max_tokens here
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        
        elif model_name == "Claude 3.7 Sonnet":
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=max_tokens,  # Added max_tokens here
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.content[0].text.strip()
        
        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt, max_tokens=max_tokens, temperature=temperature)  # Added max_tokens here
            return response.text.strip()
        
        elif model_name == "DeepSeek V3":
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review}
                ],
                max_tokens=max_tokens,  # Added max_tokens here
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        
        elif model_name == "LLaMA 3.3 70B":
            api_request_json = {
                "model": "llama3.3-70b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,  # Added max_tokens here
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
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review}
                ],
                max_tokens=max_tokens,  # Added max_tokens here
                temperature=temperature
            )
            return chat_response.choices[0].message.content.strip()
        
        elif model_name == "Grok 2":
            response = grok_client.chat.completions.create(
                model="grok-2-latest",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review}
                ],
                max_tokens=max_tokens,  # Added max_tokens here
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        
        else:
            return "Unsupported model"
    except Exception as e:
        return str(e)

# Focus on reviews that haven't been responded yet
unreplied_reviews = final_sample[final_sample["is_replied"] == "No"]

# Store results
results = []

# Iterate through unreplied reviews and generate responses
for index, row in unreplied_reviews.iterrows():
    review = row["review"]
    user_name = row["userName"]
    for model_name in models.keys():
        generated_response = generate_response(review, user_name, model_name)
        results.append({
            "Review": review,
            "LLM": model_name,
            "Generated Response": generated_response
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Print the results in a clear table format
print("\nGenerated Responses for Unreplied Reviews:")
print(tabulate(
    results_df,
    headers="keys",
    tablefmt="grid",
    showindex=False
))

# -----------------------------------------------------------------------
# Section 2- Performance Analysis
# -----------------------------------------------------------------------

# Load GPT-2 model for perplexity calculation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to calculate perplexity
def calculate_perplexity(text):
    inputs = gpt2_tokenizer (text, return_tensors="pt", max_length=512, truncation=True)
    outputs = gpt2_model (**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return float(torch.exp(loss))  # Use torch.exp to calculate perplexity

# Reinitialize the model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load similarity model for dynamic reference selection
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate sentiment alignment
def calculate_sentiment_alignment(review_sentiment, response_sentiment):
    return 1 if review_sentiment == response_sentiment else 0

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

# Function to compute sentiment
def compute_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to compute BLEU Score
def compute_bleu(reference, generated):
    reference_tokens = reference.split()
    generated_tokens = generated.split()
    return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=SmoothingFunction().method1)

# Function to compute METEOR score
def compute_meteor(reference, generated):
    return meteor_score([reference.split()], generated.split())

# Function to select the best reference response
def select_best_reference(generated_response, reference_responses):
    generated_embedding = similarity_model.encode([generated_response])
    reference_embeddings = similarity_model.encode(reference_responses)
    similarities = cosine_similarity(generated_embedding, reference_embeddings)
    best_index = similarities.argmax()
    return reference_responses[best_index]

# Define models as a dictionary
models = {
    "OpenAI GPT-4o": "gpt-4o",
    "Claude 3.7 Sonnet": "claude-3-7-sonnet-latest",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "DeepSeek V3": "deepseek-chat",
    "LLaMA 3.3 70B": "llama3.3-70b",
    "Mistral Large": "mistral-large-latest",
    "Grok 2": "grok-2-latest"
}

# Optimised prompt for better guidance
def generate_response(review, user_name, model_name, temperature=0, max_tokens=150, top_p=0.9):
    try:
        prompt = (
            f"You are a customer support agent. Write a short, empathetic, and informative response to the following customer review. "
            f"The review mentions specific issues that the customer is experiencing. Address the customer's concerns in a clear, friendly, and professional manner, "
            f"and provide suggestions or solutions where necessary. Address the customer by their name ({user_name}):\n\n"
            f"Review: {review}\n\n"
            f"Response:"
        )
        
        if model_name == "OpenAI GPT-4o":
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=1,
                stop=None,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip()
        
        elif model_name == "Claude 3.7 Sonnet":
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            )
            return response.content[0].text.strip()
        
        elif model_name == "Gemini 2.0 Flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        
        elif model_name == "DeepSeek V3":
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        
        elif model_name == "LLaMA 3.3 70B":
            api_request_json = {
                "model": "llama3.3-70b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
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
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return chat_response.choices[0].message.content.strip()
        
        elif model_name == "Grok 2":
            response = grok_client.chat.completions.create(
                model="grok-2-latest",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        
        else:
            return "Unsupported model"
    except Exception as e:
        return str(e)

# Post-processing
def post_process_response(response, user_name):
    response = response.replace("Thank you for your feedback.", f"Dear {user_name}, thank you for your feedback.")
    return response

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

# Main loop with improved parameters
results = []

for index, row in unreplied_reviews.iterrows():
    review = row["review"]
    user_name = row["userName"]
    reference_responses_list = reference_responses.get(user_name, ["No reference available"])
    review_sentiment = "positive" if compute_sentiment(review) > 0 else "negative" if compute_sentiment(review) < 0 else "neutral"
    
    for model_name in models.keys():  # Iterate over model names
        generated_response = generate_response(review, user_name, model_name, temperature=0, max_tokens=150, top_p=0.9)
        generated_response = post_process_response(generated_response, user_name)
        
        # Select the best reference response
        best_reference = select_best_reference(generated_response, reference_responses_list)
        
        # Compute automated metrics
        rouge_scores = compute_rouge(best_reference, generated_response)
        bertscore = compute_bertscore(best_reference, generated_response)
        bleu = compute_bleu(best_reference, generated_response)
        meteor = compute_meteor(best_reference, generated_response)
        perplexity = calculate_perplexity(generated_response)
        response_sentiment = "positive" if compute_sentiment(generated_response) > 0 else "negative" if compute_sentiment(generated_response) < 0 else "neutral"
        sentiment_alignment = calculate_sentiment_alignment(review_sentiment, response_sentiment)
        
        # Add results
        results.append({
            "Review": review,
            "LLM": model_name,
            "Generated Response": generated_response,
            "ROUGE-1 F1": rouge_scores["rouge-1"],
            "ROUGE-2 F1": rouge_scores["rouge-2"],
            "ROUGE-L F1": rouge_scores["rouge-l"],
            "ROUGE-L-SUM": rouge_scores["rouge-l"],  # Same as ROUGE-L for single reference
            "BERTScore": bertscore,
            "BLEU": bleu,
            "METEOR": meteor,
            "Perplexity": perplexity,
            "Sentiment Alignment": sentiment_alignment,
            "Relevance (Human)": None,
            "Coherence (Human)": None,
            "Helpfulness (Human)": None,
            "Tone (Human)": None,
            "Creativity (Human)": None
        })

# Create DataFrame
results_df = pd.DataFrame(results)

# Display the results as a table
print("\nPerformance Comparison of LLMs for Response Generation:")
print(tabulate(
    results_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))

# -----------------------------------------------------------------------
# Section 3- Time and Cost Analytics
# -----------------------------------------------------------------------

# Pricing per 1M tokens (USD)- Prining regerences in Resources section of the article
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

# Function to count tokens for OpenAI models
def count_tokens(text):
    return len(openai_tokenizer.encode(text))

# Function to calculate cost based on input and output tokens
def calculate_cost(model_name, input_tokens, output_tokens):
    input_price = PRICING[model_name]["input"] / 1_000_000  # Convert to cost per token
    output_price = PRICING[model_name]["output"] / 1_000_000  # Convert to cost per token
    total_cost = (input_tokens * input_price) + (output_tokens * output_price)
    return round(total_cost, 6)

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

# Main loop with time and cost calculation
results = []

for index, row in unreplied_reviews.iterrows():
    review = row["review"]
    user_name = row["userName"]
    reference_responses_list = reference_responses.get(user_name, ["No reference available"])
    review_sentiment = "positive" if compute_sentiment(review) > 0 else "negative" if compute_sentiment(review) < 0 else "neutral"
    
    for model_name in models.keys():  # Iterate over model names
        # Generate response and measure time/cost
        model_result = call_model(review, model_name, "response")
        generated_response = model_result[1]
        response_time = model_result[2]
        response_cost = model_result[3]
        
        # Post-process the response
        generated_response = post_process_response(generated_response, user_name)
        
        # Select the best reference response
        best_reference = select_best_reference(generated_response, reference_responses_list)
        
        # Compute automated metrics
        rouge_scores = compute_rouge(best_reference, generated_response)
        bertscore = compute_bertscore(best_reference, generated_response)
        bleu = compute_bleu(best_reference, generated_response)
        meteor = compute_meteor(best_reference, generated_response)
        perplexity = calculate_perplexity(generated_response)
        response_sentiment = "positive" if compute_sentiment(generated_response) > 0 else "negative" if compute_sentiment(generated_response) < 0 else "neutral"
        sentiment_alignment = calculate_sentiment_alignment(review_sentiment, response_sentiment)
        
        # Add results
        results.append({
            "Review": review,
            "LLM": model_name,
            "Generated Response": generated_response,
            "Response Time (s)": response_time,
            "Response Cost (USD)": response_cost,
            "ROUGE-1 F1": rouge_scores["rouge-1"],
            "ROUGE-2 F1": rouge_scores["rouge-2"],
            "ROUGE-L F1": rouge_scores["rouge-l"],
            "ROUGE-L-SUM": rouge_scores["rouge-l"],  # Same as ROUGE-L for single reference
            "BERTScore": bertscore,
            "BLEU": bleu,
            "METEOR": meteor,
            "Perplexity": perplexity,
            "Sentiment Alignment": sentiment_alignment,
        })

# Create DataFrame
results_df = pd.DataFrame(results)

# Display the results as a table
print("\nPerformance Comparison of LLMs for Response Generation:")
print(tabulate(
    results_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))

# Calculate overall averages for each LLM (only for numeric columns except time and cost)
numeric_columns_1 = [
    "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "ROUGE-L-SUM",
    "BERTScore", "BLEU", "METEOR", "Perplexity", "Sentiment Alignment"
]

# Calculate overall sum for time and cost
numeric_columns_2 = ["Response Time (s)", "Response Cost (USD)"]

# Calculate average of other metrics per LLM
average_metrics = results_df.groupby("LLM")[numeric_columns_1].mean().reset_index()

# Calculate total time and cost per LLM
total_time_cost = results_df.groupby("LLM")[numeric_columns_2].sum().reset_index()

# Merge both results on "LLM"
overall_scores = pd.merge(average_metrics, total_time_cost, on="LLM")

# Print overall performance comparison
print("\nOverall Performance Comparison of LLMs:")
print(tabulate(
    overall_scores,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))
