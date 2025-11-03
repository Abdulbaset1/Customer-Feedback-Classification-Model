import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

class SentimentAnalyzer:
    def __init__(self, model_path="sentiment_bestmodel.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping (same as in your training)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
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
sentiment_analyzer = SentimentAnalyzer()
