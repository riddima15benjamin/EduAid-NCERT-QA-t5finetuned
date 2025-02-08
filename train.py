from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

tokenizer = T5Tokenizer.from_pretrained('t5-small')   ###lightweight model, easier and simpler user interface for students
model = T5ForConditionalGeneration.from_pretrained('t5-small')

dataset = load_dataset("json", data_files="dataset.json")

def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
