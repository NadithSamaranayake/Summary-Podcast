from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, pipeline

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Function to generate conversation
def generate_conversation(summary, max_tokens=1024):
    conversation = []
    prompt = f"Party 1: Can you summarize the main points of this podcast?\nParty 2:"

    # Tokenize the prompt and ensure it stays within the model's token limit
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_tokens)
    prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    conversation.append(f"Party 1: Can you summarize the main points of this podcast?")
    
    # Generate response from Party 2
    response_party2 = generator(prompt, max_new_tokens=100)[0]['generated_text']
    conversation.append(f"Party 2: {response_party2.strip()}")

    # Adjust conversation length to fit within the max_tokens limit
    while sum(len(tokenizer.encode(c)) for c in conversation) > max_tokens:
        conversation.pop(0)  # Remove the oldest message if the conversation exceeds the token limit
    
    # Simulate further conversation between Party 1 and Party 2
    for i in range(3):  # Generate 3 more rounds of conversation
        prompt = f"Party 1: {response_party2.strip()}\nParty 2:"
        response_party2 = generator(prompt, max_new_tokens=100)[0]['generated_text']
        conversation.append(f"Party 2: {response_party2.strip()}")
        
        # Trim the conversation if it exceeds max tokens
        while sum(len(tokenizer.encode(c)) for c in conversation) > max_tokens:
            conversation.pop(0)
        
        prompt = f"Party 2: {response_party2.strip()}\nParty 1:"
        response_party1 = generator(prompt, max_new_tokens=100)[0]['generated_text']
        conversation.append(f"Party 1: {response_party1.strip()}")
        
        # Trim the conversation if it exceeds max tokens
        while sum(len(tokenizer.encode(c)) for c in conversation) > max_tokens:
            conversation.pop(0)

    # Return the full conversation
    full_conversation = "\n".join(conversation)
    return full_conversation

# Example use
summary = "AI in education is a growing field that helps personalize learning, enhances teacher support, and improves assessments."
conversation = generate_conversation(summary)
print(conversation)
