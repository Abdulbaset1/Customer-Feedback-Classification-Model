import torch
from transformers import BertTokenizer, BertForSequenceClassification
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
                # Use the correct GitHub releases URL format
                github_url = "https://github.com/Abdulbaset1/Customer-Feedback-Classification-Model/releases/tag/v1/sentiment_bestmodel.pt"
                self.download_model_from_url(github_url, model_path)
        
        # Load the trained weights with PyTorch 2.9 compatibility
        try:
            # Try with weights_only=False for compatibility
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            print("✅ Model loaded successfully with weights_only=False")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    def download_model_from_url(self, url, save_path):
        """Download model from GitHub releases"""
        try:
            # Fix the URL - the previous URL was incorrect
            # GitHub releases download URL should be in format: https://github.com/username/repo/releases/download/tag/filename
            # Remove any incorrect parts from the URL
            if "releases/tag/" in url:
                url = url.replace("releases/tag/", "releases/download/")
            
            print(f"Downloading model from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            
            print("✅ Model downloaded successfully!")
            
        except Exception as e:
            error_msg = f"Error downloading model: {e}"
            print(error_msg)
            raise Exception(error_msg)
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if not text or not text.strip():
            return "unknown", 0.0
            
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
