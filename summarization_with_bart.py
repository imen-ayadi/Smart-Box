from transformers import pipeline

def summarize_email_conditional(email_text, summarizer, min_input_length=50):
    """
    Summarizes the email if it's longer than min_input_length.
    Adjusts max_length parameter based on the length of the email.

    Args:
    - email_text (str): The text of the email to summarize.
    - min_input_length (int): Minimum length of email to apply summarization.

    Returns:
    - str: The summary of the email or the original email if below the min_input_length.
    """
    # Only summarize if the email is longer than min_input_length
    if len(email_text.split()) > min_input_length:
        # Dynamically set max_length to be about 75% of the email length, or any ratio that suits your need
        max_length = max(12, int(len(email_text.split()) * 0.75))
        summary = summarizer(email_text, max_length=max_length, min_length=5, do_sample=False)
        return summary[0]['summary_text']
    else:
        # Return the original email text if it's not long enough to require summarization
        return email_text

