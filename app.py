import streamlit as st
import torch
from sentiment_model import SentimentAnalyzer
import time

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    return SentimentAnalyzer("sentiment_bestmodel.pt")

def main():
    st.title("📊 Sentiment Analysis App")
    st.markdown("Analyze the sentiment of your text using our fine-tuned BERT model!")
    
    # Load model
    try:
        analyzer = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
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
        """)
        
        st.header("How to Use")
        st.markdown("""
        1. **Single Text**: Enter text in the text area and click 'Analyze Sentiment'
        2. **Batch Analysis**: Upload a CSV file with a 'Text' column for bulk analysis
        """)

if __name__ == "__main__":
    main()
