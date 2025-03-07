# Battle of the LLMs: Advanced Customer Review Analytics

A comprehensive comparison of leading Large Language Models (LLMs) for customer review analytics. It automates sentiment analysis, summarisation, and response generation for customer reviews while tracking performance metrics such as time and cost. Supported LLMs include OpenAI GPT-4o, Claude 3.7 Sonnet, Gemini 2.0 Flash, DeepSeek V3, LLaMA 3.3 70B, Mistral Large, and Grok 2.

## Key Features

- **Sentiment Analysis**: Classifies reviews as positive, negative, or neutral.
- **Summarisation**: Generates concise summaries of customer reviews.
- **Response Generation**: Automates empathetic and professional customer support responses.
- **Multi-LLM Support**: Compare results across multiple state-of-the-art LLMs.
- **Time and Cost Tracking**: Measures execution time and calculates costs for each task.
- **Asynchronous Execution**: Improves efficiency with concurrent API calls.
- **Customisable Prompts**: Tailor prompts for sentiment analysis, summarisation, and response generation.

## Use Cases

- Automate customer review processing for businesses.
- Benchmark and compare the performance of different LLMs.
- Optimise costs and response times for customer support workflows.

## Supported LLMs

- OpenAI GPT-4o
- Claude 3.7 Sonnet
- Gemini 2.0 Flash
- DeepSeek V3
- LLaMA 3.3 70B
- Mistral Large
- Grok 2

## Technologies Used

- Python
- OpenAI API
- Anthropic API
- Google Generative AI API
- Mistral API
- DeepSeek API
- LLaMA API
- Grok API
- Pandas, Tabulate, Asyncio, and more.

## Why This Project?

This project is designed for developers, data scientists, and businesses looking to leverage the power of LLMs for customer review analytics. By comparing multiple LLMs, it helps users make informed decisions about which model best suits their needs in terms of accuracy, speed, and cost.

## How to Use

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt` or `!python -m pip install -r requirements.txt`.
3. Set up API keys in a `.env` file.
4. Run the script to analyse customer reviews and generate insights.
5. I used Visual Studio Code for this project, but you can use any IDE that supports Python.

## Table of Scripts

| File Name                                      | Description                                                                 |
|------------------------------------------------|-----------------------------------------------------------------------------|
| `1_extract_amazon_reviews.py`                  | Extract customer reviews for Amazon Shopping Apps from Google Play and App Store for analysis.                         |
| `2_sentiment_analysis_llms.py`                | performs sentiment analysis using various Large Language Models (LLMs) and compares their performance across different datasets.        |
| `3_summarise_customer_reviews_llms.py`        | summarises customer reviews using 7 leading LLMs to generate concise insights from large volumes of feedback.        |
| `4_generate_review_responses_llms.py`         | generate responses for customer reviews that have not yet been addressed using 7 leading LLMs. |
| `5_sentiment_summary_response_e2e_llms.py`    | End-to-end pipeline for sentiment analysis, summarization, response generation, and comparison of total response time and cost for each LLM. |
| `5_sentiment_summary_response_e2e_llms_with_rate_limiter.py` | (Out of scope) Enhanced version of 5_sentiment_summary_response_e2e_llms.py with rate limiting for Mistral Large to handle API throttling issues. |
