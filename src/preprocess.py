import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Remove URLs, @mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|@\w+|#', '', str(text))
    # 2. Keep only letters and basic emotion-carrying punctuation
    text = re.sub(r'[^a-zA-Z!?\s]', '', text)
    # 3. Lowercase and strip whitespace
    text = text.lower().strip()
    return text

def run_preprocessing(input_path, output_path):
    print(" Starting Preprocessing...")
    # Loading based on your file structure: index, text, label
    df = pd.read_csv(input_path, names=['id', 'text', 'label'], header=None)
    
    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Map labels to names (Standard for this dataset)
    emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    df['emotion_name'] = df['label'].map(emotion_map)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f" Cleaned data saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing('data/text.csv', 'data/cleaned_text.csv')