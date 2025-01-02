import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline

# Load dataset
def load_and_preprocess_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Filter for relevant columns
    df = df[['question', 'answer', 'question_len', 'answer_len']]

    # Remove rows with missing or invalid data
    df = df.dropna()
    return df

# Preprocess for GPT-2
def preprocess_data(df, tokenizer):
    inputs = []
    for _, row in df.iterrows():
        input_text = f"Q: {row['question']} A: {row['answer']}"
        inputs.append(tokenizer(input_text, truncation=True, padding='max_length', max_length=128))
    return inputs

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set a padding token (use eos_token as pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess the data
file_path = "dialogs_expanded.csv"
df = load_and_preprocess_data(file_path)
preprocessed_data = preprocess_data(df, tokenizer)

# Dataset class
class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add labels for loss computation
        item["labels"] = item["input_ids"].clone()
        return item

# Create dataset
input_ids = [d["input_ids"] for d in preprocessed_data]
attention_masks = [d["attention_mask"] for d in preprocessed_data]

dataset = ConversationDataset({
    "input_ids": input_ids,
    "attention_mask": attention_masks,
})

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# Evaluate Model
def evaluate_model(chatbot, df):
    bleu_scores = []
    coherence_scores = []
    for _, row in df.iterrows():
        question = row['question']
        true_answer = row['answer']
        generated_answer = chatbot(f"Q: {question}")[0]['generated_text']

        # Compute BLEU score
        bleu_scores.append(sentence_bleu([true_answer.split()], generated_answer.split()))

        # Compute contextual coherence
        true_vector = tokenizer(true_answer, return_tensors="pt")['input_ids']
        generated_vector = tokenizer(generated_answer, return_tensors="pt")['input_ids']
        coherence_scores.append(cosine_similarity(true_vector, generated_vector).flatten()[0])

    return {
        "BLEU Score": sum(bleu_scores) / len(bleu_scores),
        "Contextual Coherence": sum(coherence_scores) / len(coherence_scores)
    }

# Load the model for inference
chatbot = pipeline("text-generation", model="./results", tokenizer=tokenizer)

# Test the model
test_data = df.sample(100)  # Use a subset for evaluation
evaluation_results = evaluate_model(chatbot, test_data)
print(evaluation_results)
