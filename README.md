 


# Smart Box: AI-Powered Email Management System
# Project Overview
Smart Box is an advanced AI-based email management system developed to enhance internal communication processes for organizations. It leverages NLP models and machine learning techniques to automate tasks such as email categorization, sentiment analysis, named entity recognition (NER), summarization, and response generation. This system significantly reduces the time spent managing emails, allowing employees to focus on more strategic tasks.

# Key Features

### Automated Email Categorization:
Automatically classifies incoming emails into predefined categories such as Urgent Requests, Client Communications, and Project Updates.
Information Extraction: Extracts key details like dates, names, and locations from emails for quick access.
Sentiment Analysis: Analyzes the tone of emails, helping prioritize critical communications based on sentiment.
Email Summarization: Summarizes lengthy emails into concise, digestible formats, improving readability and productivity.
Response Generation: Generates professional email responses, saving time and improving communication efficiency.
Technologies Used

# NLP Models:
RoBERTa, BERT, DeBERTa, T5
Frameworks & Libraries: Gradio, Hugging Face Transformers, Docker, spaCy
Programming Languages: Python
Deployment: Deployed on Hugging Face Spaces
Project Structure

fine_tuned_roberta_for_category_model_: Pre-trained model for email categorization
fine_tuned_roberta_for_sentiment_analysis: Pre-trained model for sentiment analysis
llama2_response_mail_generator.py: Generates responses using LLaMA-2
summarization_with_bart.py: Summarizes emails using BART model
app.py: Main application logic
Dockerfile: Docker setup for deployment
Setup & Installation

![Screenshot_110](https://github.com/user-attachments/assets/03ef6b55-2ae4-401d-bc54-e1ba978a472c)


# Clone the repository:
git clone https://github.com/imen-ayadi/Smart-Box.git
Install dependencies:
pip install -r requirements.txt
Run the application:
python app.py
Future Enhancements

Implement multilingual support for non-English emails.
Integration with popular email services (Gmail, Outlook) for seamless deployment.

---
title: Smart Box
emoji: 🐢
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
"# Smart-Box"
