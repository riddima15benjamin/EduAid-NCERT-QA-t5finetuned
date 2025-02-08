from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('./model')  # Path to your fine-tuned model
model = T5ForConditionalGeneration.from_pretrained('./model')

# Load the dataset
with open("final_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Create a dictionary for quick lookup
qa_dict = {item["question_code"]: item["answer"] for item in dataset}

# Function to generate an answer using question_code
def generate_answer(question_code):
    if question_code not in qa_dict:
        return "Question not found in dataset."
    
    # Get the question and the answer
    question_text = qa_dict[question_code]
    
    # Construct a more direct question-answer prompt
    input_text = f"Answer the following question: {question_text}"

    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # Generate the output with some adjustments
    with torch.no_grad():  # Disable gradients for inference
        output = model.generate(
            input_ids,
            max_length=250,  # Ensure sufficient space for the full answer
            num_beams=5,     # Beam search for improved quality
            no_repeat_ngram_size=2,  # Prevent repeating n-grams
            early_stopping=False,  # Allow full generation
            do_sample=True,     # Enable sampling for better diversity
            temperature=0.7,    # Set randomness (0-1) for diversity
            top_k=50,           # Limit sampling to top-k logits
            top_p=0.95          # Use top-p (nucleus) sampling
        )

    # Decode the generated output and return the response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Check if the answer is missing any starting part and prepend it if necessary
    if answer.startswith("Answer the following question:"):
        answer = answer[len("Answer the following question: "):]  # Remove redundant part if it appears

    return answer.strip()  # Remove any extra spaces

# Main loop to ask user for question code and generate answer
def interactive_qa():
    while True:
        input_code = input("Enter question code (or type 'exit' to quit): ").strip()
        
        if input_code.lower() == 'exit':
            print("Exiting...")
            break
        
        answer = generate_answer(input_code)
        print(f"Answer: {answer}")
        print("-" * 20)

# Run the interactive QA
interactive_qa()
