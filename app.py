import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, render_template, request

app = Flask(__name__)

tokenizer = None
model = None
qa_dict = {}

def load_model_and_data():
    global tokenizer, model, qa_dict 
    tokenizer = T5Tokenizer.from_pretrained('./model')  
    model = T5ForConditionalGeneration.from_pretrained('./model')

    with open("final_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    global qa_dict
    qa_dict = {item["question_code"]: item["answer"] for item in dataset}

load_model_and_data()  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/main_screen/<subject>', methods=['GET', 'POST'])
def main_screen(subject):
    answer = ""
    if request.method == 'POST':
        question_code = request.form['question_code'].strip()
        if question_code in qa_dict:
            input_text = f"Answer: {qa_dict[question_code]}"  # Ensure first word is not lost
            input_ids = tokenizer(input_text, return_tensors='pt').input_ids
            with torch.no_grad():
                output = model.generate(input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=2)
            
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            answer = "Question not found."
    
    return render_template('main_screen.html', answer=answer, subject=subject)

if __name__ == '__main__':
    app.run(debug=True)
