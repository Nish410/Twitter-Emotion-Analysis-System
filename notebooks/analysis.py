import pandas as pd
import plotly.express as px

def show_analysis():
    df = pd.read_csv('data/cleaned_text.csv')
    
    # 1. Class Distribution (How many Joy vs Anger?)
    fig = px.histogram(df, x='emotion_name', title='Emotion Distribution in Dataset',
                       color='emotion_name', template='plotly_dark')
    fig.show()
    
    # 2. Sequence Length Analysis
    df['text_len'] = df['cleaned_text'].str.split().str.len()
    avg_len = df['text_len'].mean()
    print(f"Average Tweet Length: {avg_len:.2f} words")

if __name__ == "__main__":
    show_analysis()