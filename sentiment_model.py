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
        
        # Load the trained weights with weights_only=False for PyTorch 2.6 compatibility
        try:
            # First try with weights_only=True (safer)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except Exception as e:
            print(f"Loading with weights_only=True failed: {e}")
            print("Trying with weights_only=False...")
            # Fallback to weights_only=False for compatibility
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        
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
                st.info("📥 Downloading model from GitHub releases...")
            
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
                st.success("✅ Model downloaded successfully!")
            
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

# Alternative loader function for better error handling
def safe_load_model(model_path, model_url=None):
    """Safely load model with PyTorch 2.6 compatibility"""
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    
    # Download if needed
    if not os.path.exists(model_path) and model_url:
        analyzer = SentimentAnalyzer()
        analyzer.download_model_from_url(model_url, model_path)
    
    # Try different loading strategies
    try:
        # Strategy 1: weights_only=True (safest)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("✅ Model loaded with weights_only=True")
    except Exception as e1:
        print(f"❌ Strategy 1 failed: {e1}")
        try:
            # Strategy 2: weights_only=False
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print("✅ Model loaded with weights_only=False")
        except Exception as e2:
            print(f"❌ Strategy 2 failed: {e2}")
            try:
                # Strategy 3: Load with specific pickle module (for older PyTorch versions)
                state_dict = torch.load(model_path, map_location=device, pickle_module=torch.pickle)
                model.load_state_dict(state_dict)
                print("✅ Model loaded with custom pickle module")
            except Exception as e3:
                print(f"❌ Strategy 3 failed: {e3}")
                raise Exception(f"All loading strategies failed: {e3}")
    
    model.to(device)
    model.eval()
    return model, tokenizer, device
