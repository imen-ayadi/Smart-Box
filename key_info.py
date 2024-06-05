import spacy
from transformers import pipeline
import re
from dateutil.parser import parse

# Regex pattern for dates
def extract_entities(email_text, nlp, ner_pipeline):
    date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:th|st|nd|rd)?,\s+\d{4}\b'
    # Use spaCy for initial extraction
    doc = nlp(email_text)
    spacy_entities = [{"Text": ent.text, "Type": ent.label_} for ent in doc.ents]

    # Use transformer model for refined extraction
    transformer_entities = ner_pipeline(email_text)
    transformer_entities = [{"Text": ent['word'], "Type": ent['entity'], "Score": ent['score']} for ent in transformer_entities if ent['score'] > 0.75]

    # Extract dates using regex
    potential_dates = re.findall(date_pattern, email_text)
    dates = [parse(date).strftime('%Y-%m-%d') for date in potential_dates]

    return {
        "spaCy Entities": spacy_entities,
        "Transformer Entities": transformer_entities,
        "Dates": dates
    }


