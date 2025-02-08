# EduAid - NCERT Textbook Solutions Web App

## About 
EduAid is a web application built using Flask that provides chapter-wise, question-specific answers for NCERT Class 11 and 12 subjects. The app uses a T5-Huggingface model for answer generation and retrieval from a pre-built custom dataset.

## Features

📚 Subject Selection: Choose from Class 11 & 12 NCERT subjects (Biology, Chemistry, Physics).

🔍 Question Code Search: Retrieve answers using structured question codes (e.g., 1.1.1).

🤖 T5-Based Answer Generation: Utilizes a fine-tuned T5 model to provide detailed responses.

🎨 Flask Web UI: A smooth and responsive UI built using Flask and Jinja templates.

⚡ Fast Lookups: Preloaded dataset for quick and efficient answer retrieval.

## Technologies Used

Python 

Flask (for web framework)

PyTorch (for model inference)

Hugging Face Transformers (for T5-based NLP processing)

JSON (for structured question-answer dataset storage)

# Installation

Prerequisites

Ensure you have Python and pip installed.
```bash
pip install flask torch transformers
```
Clone the Repository
```bash
git clone https://github.com/your-repo/eduaid.git
cd eduaid
```
Load the Model and Dataset

Ensure final_dataset.json exists in the project root.

Run the Application
```bash
python app.py  # Start the Flask web server
```
Access the Web App

Once the Flask server is running, open your browser and navigate to:
```bash
http://127.0.0.1:5000/
```
Question Code Structure

EduAid follows a structured format for retrieving answers efficiently:

X.Y.Z
X - Chapter Number
Y - Exercise Number
Z - Question Number

For example, 1.1.1 refers to Chapter 1, Exercise 1, Question 1.

