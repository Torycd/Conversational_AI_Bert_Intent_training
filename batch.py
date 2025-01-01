import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class ConversationalAIModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=5):
        """
        Initialize BERT model for intent classification
        
        Args:
            model_name (str): Pretrained BERT model
            num_labels (int): Number of intent categories
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
    
    def prepare_data(self, texts, labels, batch_size=16):
        """
        Prepare and tokenize input data
        
        Args:
            texts (list): Input conversation texts
            labels (list): Corresponding intent labels
            batch_size (int): Batch size for DataLoader
        
        Returns:
            DataLoader: Batched dataset ready for training
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_model(self, train_dataloader, val_dataloader=None, epochs=5, learning_rate=2e-5):
        """
        Train the BERT model on conversational data
        
        Args:
            train_dataloader (DataLoader): Training data
            val_dataloader (DataLoader): Validation data
            epochs (int): Number of training epochs
            learning_rate (float): Optimizer learning rate
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_steps = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                train_steps += 1
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            
            avg_train_loss = total_train_loss / train_steps
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                print(f"Validation loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pt')
    
    def evaluate(self, dataloader):
        """
        Evaluate model on validation/test data
        
        Args:
            dataloader (DataLoader): Validation/test data
        
        Returns:
            float: Average loss
        """
        self.model.eval()
        total_loss = 0
        steps = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
                steps += 1
        
        return total_loss / steps
    
    def predict_intent(self, text):
        """
        Predict intent for a given text
        
        Args:
            text (str): Input conversation text
        
        Returns:
            int: Predicted intent category
        """
        self.model.eval()
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return torch.argmax(outputs.logits, dim=1).item()

def main():
    # Load and split the SNIPS dataset
    dataset = load_dataset("snips_built_in_intents")
    full_data = dataset['train']
    train_test_split = full_data.train_test_split(test_size=0.2, seed=42)
    
    # Create validation split from training data
    train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)
    
    # Prepare datasets
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    test_data = train_test_split['test']
    
    # Initialize model
    num_labels = len(set(train_data['label']))
    ai_model = ConversationalAIModel(num_labels=num_labels)
    
    # Prepare data loaders
    train_dataloader = ai_model.prepare_data(train_data['text'], train_data['label'])
    val_dataloader = ai_model.prepare_data(val_data['text'], val_data['label'])
    test_dataloader = ai_model.prepare_data(test_data['text'], test_data['label'])
    
    # Train model
    ai_model.train_model(train_dataloader, val_dataloader)
    
    # Evaluate on test set
    test_predictions = []
    test_labels = []
    
    ai_model.model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = [b.to(ai_model.device) for b in batch]
            outputs = ai_model.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    print("\nTest Set Performance:")
    print(classification_report(test_labels, test_predictions))

if __name__ == "__main__":
    main()
    


