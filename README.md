# Battle of the LLMs: Advanced Customer Review Analytics
![Logo](https://github.com/user-attachments/assets/85b62b4d-55a7-44fc-9e38-21bc3047caa5)


A comprehensive comparison of leading Large Language Models (LLMs) for customer review analytics. It automates sentiment analysis, summarisation, and response generation for customer reviews while tracking performance metrics such as time and cost. Supported LLMs include OpenAI GPT-4o, Claude 3.7 Sonnet, Gemini 2.0 Flash, DeepSeek V3, LLaMA 3.3 70B, Mistral Large, and Grok 2.


## Key Features

### **Core Functionality**
- **Sentiment Analysis**: Classifies reviews as positive, negative, or neutral.  
- **Summarisation**: Generates concise summaries of customer reviews.  
- **Generating User-Centric Responses**: Incorporates user names and empathetic language for enhanced customer support.  

### **Multi-LLM Integration**
- **Multi-API Integration**: Seamlessly integrates multiple LLMs (OpenAI, Anthropic, Google, Mistral, DeepSeek, Meta, XAI).  
- **Multi-LLM Support**: Fair comparison across multiple state-of-the-art LLMs.  

### **Efficiency & Performance**
- **Asynchronous Execution**: Improves efficiency with concurrent API calls using asyncio.  
- **Time and Cost Tracking**: Measures execution time and calculates costs for each task.  
- **Dynamic Token Counting**: Implements token counting (tiktoken) for efficient API usage.  

### **Error Handling & Reliability**
- **Robust Error Handling**: Implements exception handling for graceful degradation and fallback mechanisms.  
- **Sentiment Validation**: Ensures outputs are valid (positive, negative, or neutral).  

### **Security & Configuration**
- **Secure API Key Management**: Uses environment variables for secure and configurable API key handling.  

### **Customisation & Scalability**
- **Customisable Prompts**: Tailor prompts for sentiment analysis, summarisation, and response generation.  
- **Scalable Design**: Modular structure allows easy addition of new models or tasks.  

### **Output & Reporting**
- **Text Formatting**: Ensures clean, readable output with text wrapping for long responses.  
- **Comparative Analysis**: Generates tables for model performance, cost, and time efficiency insights.  

### **End-to-End Workflow**
- **End-to-End Pipeline**: Orchestrates a complete workflow from sentiment analysis to response generation.  



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
  ![Models](https://github.com/user-attachments/assets/1c8bf838-e902-42f0-87ff-5b311308acec)



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
![Overview](https://github.com/user-attachments/assets/b601f0cc-1360-4d82-a2ef-a2910ee78e9f)



## How to Use

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt` or `!python -m pip install -r requirements.txt`.
3. Set up API keys in a `.env` file.
4. Run the script to analyse customer reviews and generate insights.
5. I used Visual Studio Code with Python 3.13.2 64-bit for this project, but you can use any IDE that supports Python.


## Table of Scripts

| File Name                                      | Description                                                                 |
|------------------------------------------------|-----------------------------------------------------------------------------|
| `1_extract_amazon_reviews.py`                  | Extracts customer reviews for Amazon Shopping Apps from Google Play and App Store for analysis.                         |
| `2_sentiment_analysis_llms.py`                | Performs sentiment analysis using various Large Language Models (LLMs) and compares their performance across different datasets.        |
| `3_summarise_customer_reviews_llms.py`        | Summarises customer reviews using 7 leading LLMs to generate concise insights from large volumes of feedback and compares LLMs performance.        |
| `4_generate_review_responses_llms.py`         | Generates responses for customer reviews that have not yet been addressed using 7 leading LLMs and compares LLMs performance. |
| `5_sentiment_summary_response_e2e_llms.py`    | End-to-end pipeline for sentiment analysis, summarisation, response generation, and comparison of total response time and cost for each LLM. |
