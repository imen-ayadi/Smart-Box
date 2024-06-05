import gradio as gr
import pandas as pd
from key_info import extract_entities
from summarization_with_bart import summarize_email_conditional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import spacy

"""**Original code**

**CSS for Interface**
"""

custom_css = ''' @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css');

body {
    background-color: #eef1f5; /* Light grey-blue background for a neutral, clean look */
}
label {
    color: #34495e; /* Dark blue-grey for a professional appearance */
    font-weight: bold;
}
textarea, input, select, button {
    background-color: #ffffff; /* Crisp white background for input fields and buttons */
    border: 1px solid #bdc3c7; /* Soft grey border for a subtle, refined look */
    color: #2c3e50; /* Darker shade of blue-grey for text, enhancing readability */
}
button {
    background-color: #3498db; /* Bright blue for buttons to stand out */
    color: black ; /* White text on buttons for clarity */
    border-radius: 4px; /* Slightly rounded corners for a modern touch */
    font-weight: bold; /* Bold text for emphasis */
    font-size: 16px; /* Sizable text for easy interaction */
}
button[type="submit"], button[type="reset"], button[type="button"] {
    font-weight: bold; /* Ensures all actionable buttons are prominent */
    font-size: 18px; /* Larger text size for better visibility and impact */
}
.result-box {
    background-color: #ecf0f1; /* Very light grey for result boxes, ensuring focus */
    color: #2c3e50; /* Consistent dark blue-grey text for uniformity */
    border: 1px solid #bdc3c7; /* Matching the input field borders for design coherence */
}
.gradio-toolbar {
    background-color: #ffffff; /* Maintains a clean, unobtrusive toolbar appearance */
    border-top: 2px solid #3498db; /* A pop of bright blue to delineate the toolbar */
}

'''

"""**Seperate** **Interface**"""
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
model_path = './fine_tuned_roberta_for_category_model_'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load model and tokenizer from the drive
model_sentiment_path = './fine_tuned_roberta_for_sentiment_analysis_2000_'
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_sentiment_path)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_sentiment_path)
model_sentiment.eval()
model_sentiment.to(device)


model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
model_basename = "llama-2-7b-chat.Q2_K.gguf"  # The model is in bin format

# Download the model file
print('downloading llama model...')
model_path_llama = hf_hub_download(repo_id=model_name_or_path, filename=model_basename, force_download=True, local_dir="./llama_model")
print('finished download...')
# Initialize the Llama model with appropriate settings for GPU
lcpp_llm = Llama(
    model_path=model_path_llama,
)

def generate_email_response(email_prompt):
    # Check input received by the function
    print("Received prompt:", email_prompt)

    # Determine if the input is a shorthand command or an actual email
    if 'email to' in email_prompt.lower():
        # Assume it's a shorthand command, format appropriately
        formatted_prompt = f'''
        Email received: "{email_prompt}"
        Respond to this email, ensuring a professional tone, providing a concise update, and addressing any potential concerns the sender might have.
        Response:
        '''
    else:
        # Assume it's direct email content
        formatted_prompt = f'''
        Email received: "{email_prompt}"
        Respond to this email, ensuring a professional tone, providing a concise update, and addressing any potential concerns the sender might have.
        Response:
        '''

    # Generate response using Llama-2 model
    try:
        response = lcpp_llm(
            prompt=formatted_prompt,
            max_tokens=256,
            temperature=0.5,
            top_p=0.95,
            repeat_penalty=1.2,
            top_k=150,
            echo=True
        )
        generated_response = response["choices"][0]["text"]
        # Remove the input part from the output if it is included
        if formatted_prompt in generated_response:
            generated_response = generated_response.replace(formatted_prompt, '').strip()
        print("Generated response:", generated_response)
        return generated_response
    except Exception as e:
        print("Error in response generation:", str(e))
        return "Failed to generate response, please check the console for errors."
    
def classify_sentiment(text):
    # Encode the text using the tokenizer
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model_sentiment(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Convert predictions to probabilities and sentiment category
    probabilities = predictions.cpu().numpy()[0]
    categories = ["Positive", "Neutral", "Negative"]
    predicted_sentiment = categories[probabilities.argmax()]

    # Return the predicted sentiment and the confidence
    confidence = max(probabilities)
    return f"Sentiment: {predicted_sentiment}, Confidence: {confidence:.2f}"

def generate_summary(email_text):
    return summarize_email_conditional(email_text, summarizer)

def display_entities(email_text):
    try:
        results = extract_entities(email_text, nlp, ner_pipeline)

        # Convert to DataFrames
        data_spacy = pd.DataFrame(results['spaCy Entities'])
        data_transformer = pd.DataFrame(results['Transformer Entities'])

        return data_spacy, data_transformer, ", ".join(results['Dates'])
    except Exception as e:
        print(f"Error: {e}")
        # Return empty outputs in case of error
        return pd.DataFrame(), pd.DataFrame(), ""
    
def classify_email(email):
    # Encode the email text using the tokenizer
    inputs = tokenizer(email, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Convert predictions to probabilities and category
    probabilities = predictions.cpu().numpy()[0]
    categories = ["Urgent Requests", "Project Updates", "Client Communications", "Meeting Coordination", "Internal Announcements"]
    predicted_category = categories[probabilities.argmax()]

    # Return the predicted category and the confidence
    confidence = max(probabilities)
    return f"Category: {predicted_category}, Confidence: {confidence:.2f}"


iface_category = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(label="Email",lines=10, placeholder="Enter Email Content Here..."),
    outputs="text",
    title="Email Category Classifier",
    stop_btn=gr.Button("Stop", variant="stop", visible=True),
    description="This model classifies email text into one of five categories: Urgent Requests, Project Updates, Client Communications, Meeting Coordination, Internal Announcements."
)


iface_sentiment = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(label="Email", lines=5, placeholder="Enter Email Text Here..."),
    outputs=gr.Textbox(label="Sentiment Analysis"),
    stop_btn=gr.Button("Stop", variant="stop", visible=True),
    title="Sentiment Analysis"
)


iface_summary = gr.Interface(
    fn=generate_summary,
    inputs=[gr.Textbox(lines=5, placeholder="Enter Email Text Here...")],
    outputs=gr.Textbox(label="Generated Summary"),
    stop_btn=gr.Button("Stop", variant="stop", visible=True),
    title="Summary Generation"
)

iface_ner = gr.Interface(
    fn=display_entities,
    inputs=gr.Textbox(label="Email", lines=5, placeholder="Enter Email Text Here..."),
    outputs=[
        gr.Dataframe(label="spaCy Entity Recognition"),
        gr.Dataframe(label="Transformer Entity Recognition"),
        gr.Textbox(label="Extracted Dates")
    ],
    stop_btn=gr.Button("Stop", variant="stop", visible=True),
    title="NER Analysis",
    description="Performs Named Entity Recognition using spaCy and Transformer models."
)
iface_response = gr.Interface(
    fn=generate_email_response,
    inputs=gr.Textbox(label="Email", lines=10, placeholder="Enter the email prompt..."),
    outputs=gr.Textbox(label="Generated Email Response"),
    title="Email Response Generator",
    stop_btn=gr.Button("Stop", variant="stop", visible=True),
    description="Generate email responses using Llama-2 model."
)

# Using tabs to organize the interfaces
tabs = gr.TabbedInterface([iface_category, iface_sentiment,iface_summary,iface_ner,iface_response], ["Category", "Sentiment"," Summary","NER","Response Generator"], css=custom_css)
tabs.launch(share=True)

