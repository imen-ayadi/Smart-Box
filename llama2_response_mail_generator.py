from huggingface_hub import hf_hub_download

from llama_cpp import Llama

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"  # The model is in bin format

# Download the model file
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Initialize the Llama model with appropriate settings for GPU
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,  # CPU cores to use
    n_batch=512,  # Batch size for processing; adjust as per your VRAM capacity
    n_gpu_layers=32  # Number of layers to run on GPU, dependent on your GPU's VRAM
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