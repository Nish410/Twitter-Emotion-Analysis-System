import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 1. Configuration
MODEL_NAME = "microsoft/deberta-v3-small"
DATA_PATH = "data/text.csv"
SAVE_PATH = "./models/final_model"

# 2. Custom Dataset Class (No external 'datasets' library needed)
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def start_training():
    print(" Loading and preparing data...")
    # Load data with low_memory=False to avoid the mixed-type warning
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # --- SPEED UP: Use a subset of 2,000 rows so your Mac doesn't hang ---
    df = df.sample(n=2000, random_state=42) 
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].astype(str).tolist(), 
        df['label'].astype(int).tolist(), 
        test_size=0.2
    )

    # 3. Tokenization
    print(f" Downloading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)

    # 4. Model Setup
    print(" Initializing DeBERTa model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)

    # 5. Training Arguments (Fixed for latest Transformers version)
    training_args = TrainingArguments(
        output_dir="./models/emotion_checkpoints",
        eval_strategy="epoch",        # Fixed name
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Low batch size to save RAM
        per_device_eval_batch_size=4,
        num_train_epochs=1,            # 1 Epoch is enough for a project demo
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 7. Start Training
    print(" Training started! (This should take ~5-10 mins on Mac)")
    trainer.train()

    # 8. Save Final Model
    print(f"Saving model to {SAVE_PATH}...")
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    

if __name__ == "__main__":
    start_training()