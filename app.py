import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Page Configuration - Neutral Icon
st.set_page_config(page_title="Sentira AI | Emotion Analysis", page_icon="📊", layout="wide")

# Initialize Session State
if 'tweet_text' not in st.session_state:
    st.session_state.tweet_text = ""

# 2. Model Loading
@st.cache_resource
def load_model():
   
    model_path = "Nish40/emotion-deberta-v3" 
    
    # 2. Add 'subfolder="final_model"' because that is where your files are on HF
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="final_model", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, subfolder="final_model")
    
    return tokenizer, model
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# 3. Sidebar - Minimalist & Professional
with st.sidebar:
    st.title("Sentira AI Engine")
    st.markdown("---")
    st.markdown("### System Metadata")
    st.code("System Ref: ML-TED-39", language="bash") 
    st.markdown("**Version:** 1.0.4 (Stable)")
    st.markdown("---")
    st.markdown("**Technical Architecture**")
    st.info("Utilizing the DeBERTa-v3 transformer model for high-precision emotion mapping in short-form text.")

# 4. Main UI
st.title("Advanced Emotion Recognition Dashboard")
st.write("Real-time emotional intelligence analysis for Twitter Emotion Detection.")

# --- INPUT SECTION ---
st.subheader("Input Stream")

# Quick Template Buttons
col1, col2, col3 = st.columns(3)
if col1.button("Sample: Positive"): st.session_state.tweet_text = "I am so incredibly happy with these results!"
if col2.button("Sample: Negative"): st.session_state.tweet_text = "I am so annoyed that the system is lagging today."
if col3.button("Sample: Surprise"): st.session_state.tweet_text = "I can't believe how fast this model actually is!"

# Text Area
tweet_input = st.text_area("Enter text for analysis:", value=st.session_state.tweet_text, height=100)

# Analysis Button
analyze_btn = st.button("RUN ANALYSIS", use_container_width=True, type="primary")

st.divider()

# --- RESULTS SECTION ---
if analyze_btn:
    if not tweet_input.strip():
        st.warning("Please enter text or select a sample above.")
    else:
        tokenizer, model = load_model()
        if model is None:
            st.error("Model files not detected. Ensure the 'models/final_model' directory exists.")
        else:
            with st.spinner("Processing..."):
                cleaned = clean_text(tweet_input)
                inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten().tolist()
                
                # Emojis KEPT ONLY for Emotions as requested
                emotions = ['Sadness 😢', 'Joy 😊', 'Love ❤️', 'Anger 😡', 'Fear 😨', 'Surprise 😲']
                results_df = pd.DataFrame({"Emotion": emotions, "Confidence": probs})
                top_row = results_df.loc[results_df['Confidence'].idxmax()]

                # Display
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.subheader("Classification Results")
                    st.metric(label="Primary Emotion", value=top_row['Emotion'], 
                              delta=f"{top_row['Confidence']:.1%} confidence")
                    
                    fig = px.bar(results_df, x='Emotion', y='Confidence', color='Emotion',
                                 title="Emotional Distribution Profile")
                    st.plotly_chart(fig, use_container_width=True)

                with res_col2:
                    st.subheader("Linguistic Analysis")
                    if len(cleaned.split()) > 0:
                        wc = WordCloud(background_color="white", width=400, height=300).generate(cleaned)
                        fig_wc, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                        st.pyplot(fig_wc)