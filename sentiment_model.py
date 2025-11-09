import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import requests
import os
from pathlib import Path

class SentimentAnalyzer:
    def __init__(self, model_url=None, model_path="sentiment_bestmodel.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        
        # Download model from GitHub releases if not present locally
        if not os.path.exists(model_path):
            if model_url:
                self.download_model_from_url(model_url, model_path)
            else:
                # Default GitHub releases URL (update with your actual URL)
                github_url = "https://github.com/Abdulbaset1/Customer-Feedback-Classification-Model/releases/tag/v1/sentiment_bestmodel.pt"
                self.download_model_from_url(github_url, model_path)
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping (same as in your training)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    def download_model_from_url(self, url, save_path):
        """Download model from GitHub releases"""
        try:
            st = None
            # Try to import streamlit for progress tracking
            try:
                import streamlit as st
            except ImportError:
                pass
            
            if st:
                st.info("Downloading model from GitHub releases...")
            
            print(f"Downloading model from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as file:
                if st:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0 and st:
                            progress = downloaded_size / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded {downloaded_size}/{total_size} bytes")
            
            if st:
                st.success("Model downloaded successfully!")
            
            print("Model downloaded successfully!")
            
        except Exception as e:
            error_msg = f"Error downloading model: {e}"
            if st:
                st.error(error_msg)
            raise Exception(error_msg)
    
    def predict(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get the predicted class and confidence
        confidence, predicted_class = torch.max(predictions, dim=1)
        
        return self.label_map[predicted_class.item()], confidence.item()

# Create a global instance (optional, you can also create it in the app)
# sentiment_analyzer = SentimentAnalyzer()
