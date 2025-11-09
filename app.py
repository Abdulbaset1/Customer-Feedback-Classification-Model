import streamlit as st
import torch
from sentiment_model import SentimentAnalyzer, safe_load_model
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Initialize the model with better error handling
@st.cache_resource
def load_model():
    # Replace with your actual GitHub releases URL
    GITHUB_MODEL_URL = "https://github.com/Abdulbaset1/Customer-Feedback-Classification-Model/releases/tag/v1/sentiment_bestmodel.pt"
    
    try:
        # Try the main approach first
        return SentimentAnalyzer(model_url=GITHUB_MODEL_URL, model_path="sentiment_bestmodel.pt")
    except Exception as e:
        st.error(f"Standard loading failed: {e}")
        st.info("Trying alternative loading method...")
        
        # Fallback to safe loading
        try:
            model, tokenizer, device = safe_load_model(
                "sentiment_bestmodel.pt", 
                GITHUB_MODEL_URL
            )
            
            # Create a custom analyzer instance
            class FallbackAnalyzer:
                def __init__(self, model, tokenizer, device):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = device
                    self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
                
                def predict(self, text):
                    inputs = self.tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=128,
                        return_tensors="pt"
                    )
                    
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    confidence, predicted_class = torch.max(predictions, dim=1)
                    return self.label_map[predicted_class.item()], confidence.item()
            
            return FallbackAnalyzer(model, tokenizer, device)
            
        except Exception as fallback_error:
            st.error(f"All loading methods failed: {fallback_error}")
            raise fallback_error

def main():
    st.title("📊 Sentiment Analysis App")
    st.markdown("Analyze the sentiment of your text using our fine-tuned BERT model!")
    
    # Model loading section
    st.sidebar.header("Model Configuration")
    
    # Option to input custom GitHub URL
    github_url = st.sidebar.text_input(
        "GitHub Model URL:",
        value="https://github.com/your-username/your-repo/releases/download/v1/sentiment_bestmodel.pt",
        help="Paste the direct download URL from your GitHub releases"
    )
    
    # Load model
    try:
        with st.spinner("🔄 Loading model (PyTorch 2.6 compatibility mode)..."):
            analyzer = load_model()
        
        st.success("✅ Model loaded successfully!")
        
        # Show model info
        st.sidebar.success("**Model Status:** Loaded")
        st.sidebar.info(f"**PyTorch Version:** {torch.__version__}")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
        # Detailed troubleshooting
        st.subheader("🔧 Troubleshooting Steps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Option 1: Update Model File**
            - Re-save your model with PyTorch 2.6+
            - Use `torch.save(model.state_dict(), 'model.pt', weights_only=True)`
            """)
            
        with col2:
            st.markdown("""
            **Option 2: Manual Upload**
            - Upload the model file directly
            - Ensure it's from a trusted source
            """)
        
        # Manual upload fallback
        st.subheader("Upload Model File Manually")
        uploaded_model = st.file_uploader("Upload sentiment_bestmodel.pt", type=['pt'])
        
        if uploaded_model is not None:
            try:
                with open("sentiment_bestmodel.pt", "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                st.success("Model uploaded! Retrying load...")
                st.rerun()
                
            except Exception as upload_error:
                st.error(f"Upload failed: {upload_error}")
        return
    
    # Rest of your app code remains the same...
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Single Text Analysis", "Batch Analysis"])
    
    with tab1:
        st.header("Analyze Single Text")
        
        # Text input
        text_input = st.text_area(
            "Enter your text here:",
            placeholder="Type your text for sentiment analysis...",
            height=100
        )
        
        # Analyze button
        if st.button("Analyze Sentiment", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    time.sleep(0.5)
                    
                    # Get prediction
                    sentiment, confidence = analyzer.predict(text_input)
                    
                    # Display results
                    st.subheader("Results:")
                    
                    if sentiment == "positive":
                        st.success(f"**Sentiment:** {sentiment.upper()} 😊")
                    elif sentiment == "negative":
                        st.error(f"**Sentiment:** {sentiment.upper()} 😔")
                    else:
                        st.info(f"**Sentiment:** {sentiment.upper()} 😐")
                    
                    st.metric("Confidence", f"{confidence:.2%}")
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                    
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with a 'Text' column",
            type=['csv'],
            help="Your CSV file should have a column named 'Text' containing the texts to analyze"
        )
        
        if uploaded_file is not None:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                
                if 'Text' not in df.columns:
                    st.error("The uploaded CSV must contain a 'Text' column.")
                else:
                    st.subheader("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    if st.button("Analyze All Texts", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        sentiments = []
                        confidences = []
                        
                        for i, text in enumerate(df['Text']):
                            if pd.notna(text) and str(text).strip():
                                sentiment, confidence = analyzer.predict(str(text))
                                sentiments.append(sentiment)
                                confidences.append(confidence)
                            else:
                                sentiments.append("unknown")
                                confidences.append(0.0)
                            
                            progress = (i + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {i + 1}/{len(df)} texts...")
                        
                        df['Sentiment'] = sentiments
                        df['Confidence'] = confidences
                        
                        status_text.text("✅ Analysis complete!")
                        st.subheader("Analysis Results:")
                        st.dataframe(df)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        st.subheader("Summary Statistics:")
                        sentiment_counts = df['Sentiment'].value_counts()
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Positive", sentiment_counts.get('positive', 0))
                        with col2:
                            st.metric("Neutral", sentiment_counts.get('neutral', 0))
                        with col3:
                            st.metric("Negative", sentiment_counts.get('negative', 0))
            
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
