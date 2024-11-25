from PyPDF2 import PdfReader
from transformers import pipeline
from gtts import gTTS  # Updated import for gTTS

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Summarize text using Hugging Face BART model
def summarize_text(text, max_input_length=1024):
    # Load the summarizer model from Hugging Face (using CPU)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    
    # Split the text into chunks if it exceeds the token limit
    # We are using a simple split based on the number of characters here, but more sophisticated token-based splitting can be applied
    chunks = [text[i:i + max_input_length] for i in range(0, len(text), max_input_length)]
    
    summarized_text = ""
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False, truncation=True)
        summarized_text += summary[0]['summary_text'] + " "  # Concatenate summaries from all chunks
    
    return summarized_text

# Step 3: Generate conversation using Hugging Face GPT-2 model
def generate_conversation(summary):
    # Load GPT-2 for text generation (or any similar model like GPT-Neo)
    model_name = "gpt2"  # You can try gpt2-medium, gpt2-large, or gpt-neo
    generator = pipeline("text-generation", model=model_name, device=-1)  # Use CPU (device=-1)
    
    prompt = f"Create a friendly podcast-style conversation based on this summary: {summary}"
    conversation = generator(prompt, max_new_tokens=200, num_return_sequences=1, truncation=True)
    return conversation[0]['generated_text']

# Step 4: Convert to speech using gTTS
def text_to_speech(conversation, output_file):
    tts = gTTS(text=conversation, lang='en', slow=False)  # Create gTTS object
    tts.save(output_file)  # Save audio to file

# Full Pipeline Execution
pdf_path = "./docs/ML1a.pdf"  # Path to your PDF document
output_audio_file = "output_audio.mp3"  # Audio output file path

# Execute the steps
text = extract_text_from_pdf(pdf_path)  # Step 1: Extract text from the PDF
summary = summarize_text(text)  # Step 2: Summarize the extracted text
conversation = generate_conversation(summary)  # Step 3: Generate conversation
text_to_speech(conversation, output_audio_file)  # Step 4: Convert the conversation to speech

print(f"Podcast audio generated and saved to {output_audio_file}")
