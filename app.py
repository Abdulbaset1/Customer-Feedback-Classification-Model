import streamlit as st
import torch
from sentiment_model import SentimentAnalyzer
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    # You can specify your GitHub releases URL here
    # Replace with your actual GitHub releases URL
    GITHUB_MODEL_URL = "https://github.com/Abdulbaset1/Customer-Feedback-Classification-Model/releases/tag/v1/sentiment_bestmodel.pt"
    
    # Alternative: If you want to use the raw URL (if you've uploaded to releases)
    # GITHUB_MODEL_URL = "https://github.com/your-username/your-repo/releases/download/v1/sentiment_bestmodel.pt"
    
    return SentimentAnalyzer(model_url=GITHUB_MODEL_URL, model_path="sentiment_bestmodel.pt")

def main():
    st.title("📊 Sentiment Analysis App")
    st.markdown("Analyze the sentiment of your text using our fine-tuned BERT model!")
    
    # Model loading section
    st.sidebar.header("Model Configuration")
    
    # Option to input custom GitHub URL
    github_url = st.sidebar.text_input(
        "GitHub Model URL (optional):",
        value="https://github.com/your-username/your-repo/releases/download/v1/sentiment_bestmodel.pt",
        help="Paste the direct download URL from your GitHub releases"
    )
    
    # Load model
    try:
        with st.spinner("🔄 Loading model..."):
            if github_url and github_url != "https://github.com/your-username/your-repo/releases/download/v1/sentiment_bestmodel.pt":
                analyzer = SentimentAnalyzer(model_url=github_url, model_path="sentiment_bestmodel.pt")
            else:
                analyzer = load_model()
        
        st.success("✅ Model loaded successfully!")
        
        # Show model info
        st.sidebar.success(f"**Model Source:** {'GitHub Releases' if github_url else 'Local'}")
        if github_url:
            st.sidebar.info(f"**URL:** {github_url}")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 **Troubleshooting tips:**")
        st.markdown("""
        1. Make sure your GitHub releases URL is correct
        2. Ensure the model file is publicly accessible
        3. Check your internet connection
        4. Alternatively, upload the model file manually
        """)
        
        # Fallback: Allow manual upload
        st.subheader("Alternative: Upload Model File")
        uploaded_model = st.file_uploader("Upload your sentiment_bestmodel.pt file", type=['pt'])
        
        if uploaded_model is not None:
            with open("sentiment_bestmodel.pt", "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.success("Model uploaded successfully! Please refresh the page.")
            return
        else:
            return
    
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
                    # Add a small delay to show the spinner
                    time.sleep(0.5)
                    
                    # Get prediction
                    sentiment, confidence = analyzer.predict(text_input)
                    
                    # Display results
                    st.subheader("Results:")
                    
                    # Color-coded sentiment display
                    if sentiment == "positive":
                        st.success(f"**Sentiment:** {sentiment.upper()} 😊")
                    elif sentiment == "negative":
                        st.error(f"**Sentiment:** {sentiment.upper()} 😔")
                    else:
                        st.info(f"**Sentiment:** {sentiment.upper()} 😐")
                    
                    # Confidence meter
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                    
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Analysis")
        
        # File upload for batch processing
        uploaded_file = st.file_uploader(
            "Upload a CSV file with a 'Text' column",
            type=['csv'],
            help="Your CSV file should have a column named 'Text' containing the texts to analyze"
        )
        
        if uploaded_file is not None:
            try:
                import pandas as pd
                
                # Read the CSV file
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
                        
                        # Process each text
                        for i, text in enumerate(df['Text']):
                            if pd.notna(text) and str(text).strip():
                                sentiment, confidence = analyzer.predict(str(text))
                                sentiments.append(sentiment)
                                confidences.append(confidence)
                            else:
                                sentiments.append("unknown")
                                confidences.append(0.0)
                            
                            # Update progress
                            progress = (i + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {i + 1}/{len(df)} texts...")
                        
                        # Add results to dataframe
                        df['Sentiment'] = sentiments
                        df['Confidence'] = confidences
                        
                        status_text.text("✅ Analysis complete!")
                        
                        # Show results
                        st.subheader("Analysis Results:")
                        st.dataframe(df)
                        
                        # Download button for results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Show summary statistics
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
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a fine-tuned BERT model to analyze sentiment in text.
        
        **Sentiment Labels:**
        - 😊 Positive
        - 😐 Neutral  
        - 😔 Negative
        
        **Model Info:**
        - Base Model: BERT-base-uncased
        - Fine-tuned on custom dataset
        - 3-class classification
        - Loaded from GitHub Releases
        """)
        
        st.header("How to Use")
        st.markdown("""
        1. **Single Text**: Enter text in the text area and click 'Analyze Sentiment'
        2. **Batch Analysis**: Upload a CSV file with a 'Text' column for bulk analysis
        """)
        
        # Model status
        st.header("Model Status")
        if os.path.exists("sentiment_bestmodel.pt"):
            file_size = os.path.getsize("sentiment_bestmodel.pt") / (1024 * 1024)
            st.success(f"✅ Model loaded ({file_size:.1f} MB)")
        else:
            st.warning("❌ Model file not found")

if __name__ == "__main__":
    main()
