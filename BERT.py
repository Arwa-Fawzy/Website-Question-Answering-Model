import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import matplotlib.pyplot as plt
import numpy as np

# 1. Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. Load SQuAD dataset (built-in QA dataset)
dataset = load_dataset("arabic_squad")

# 3. Tokenize dataset
def preprocess_function(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./bert-qa-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 5. Accuracy Metric
metric = load_metric("accuracy")
accuracies = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    accuracies.append(acc["accuracy"])
    return acc

# 6. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # Use subset for quick training
    eval_dataset=tokenized_datasets["validation"].select(range(200)),
    compute_metrics=compute_metrics,
)

# 7. Train
trainer.train()

# 8. Plot Accuracy Graph
plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.grid(True)
plt.show()
