# ============================================================
# AI LECTURE PIPELINE: AUDIO -> WHISPER TRANSCRIPT -> TOPICS -> SENTIMENT
# ============================================================
# This script is designed to run in Google Colab.
# It does the following:
#   1. Install required libraries
#   2. Upload an audio file (e.g., a recorded class)
#   3. Transcribe the audio using Whisper (local, open-source)
#   4. Run topic modeling on the transcript (LDA)
#   5. Run sentiment analysis on transcript chunks
#
# You can upload this script to GitHub and your friend can
# copy-paste it into a Colab notebook.

# ============================================================
# STEP 1: INSTALL DEPENDENCIES (COLAB ONLY)
# ============================================================
# In a normal Python environment you would put these in requirements.txt
# In Colab, we install them directly with pip.

!pip install -q openai-whisper transformers sentencepiece scikit-learn

# "torch" is usually pre-installed in Colab. If not, uncomment:
# !pip install -q torch

# ============================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================

import os
from google.colab import files  # For file upload in Colab

import whisper                   # For speech-to-text
from transformers import pipeline # For sentiment analysis

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ============================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================

def upload_audio_file():
    """
    Opens a file upload dialog in Colab and returns the local filename.
    Assumes the user uploads exactly one audio file.
    Supported formats (for Whisper): mp3, m4a, wav, etc.
    """
    print("Please select your audio file (recorded class)...")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No file uploaded.")
    
    # Get the first (and only) uploaded file name
    filename = list(uploaded.keys())[0]
    print(f"Uploaded file: {filename}")
    return filename


def load_whisper_model(model_name: str = "base"):
    """
    Loads a Whisper model.
    Available sizes: tiny, base, small, medium, large
    Larger models are more accurate but slower.
    """
    print(f"Loading Whisper model: {model_name} (this can take a bit)...")
    model = whisper.load_model(model_name)
    print("Model loaded.")
    return model


def transcribe_audio(model, audio_path: str, language: str = None) -> str:
    """
    Transcribes the given audio file using Whisper.
    
    Parameters:
        model:      A loaded Whisper model
        audio_path: Path to the audio file
        language:   Optional ISO language code, e.g. 'en' for English.
                    If None, Whisper will try to detect the language.
    
    Returns:
        transcript (str): The full transcribed text.
    """
    print(f"Transcribing audio file: {audio_path} ...")
    result = model.transcribe(audio_path, language=language)
    transcript = result.get("text", "").strip()
    print("Transcription completed.")
    return transcript


def save_transcript(text: str, filename: str = "transcript.txt") -> str:
    """
    Saves the transcript to a text file.
    
    Parameters:
        text:      Transcript string
        filename:  Output filename
    
    Returns:
        The path of the saved file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcript saved to: {filename}")
    return filename


def chunk_text(text: str, chunk_size: int = 120) -> list:
    """
    Splits a long text into smaller chunks of roughly 'chunk_size' words.
    This is useful for both topic modeling and sentiment analysis.
    
    Parameters:
        text:        Full transcript
        chunk_size:  Number of words per chunk
    
    Returns:
        List of text chunks (each chunk is a short "document").
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def run_topic_modeling(
    documents: list,
    n_topics: int = 5,
    max_features: int = 1000,
    max_df: float = 0.95,
    min_df: int = 2
):
    """
    Runs simple topic modeling (LDA) on a list of short documents (chunks).
    
    Parameters:
        documents:    List of text chunks
        n_topics:     Number of topics to extract
        max_features: Maximum vocabulary size
        max_df:       Ignore words that appear in more than this fraction of documents
        min_df:       Ignore words that appear in fewer than this number of documents
    
    Returns:
        lda_model:    Fitted LDA model
        vectorizer:   Fitted CountVectorizer
        doc_term_mat: Document-term matrix
    """
    if len(documents) < 2:
        print("Not enough documents for topic modeling. Need at least 2 chunks.")
        return None, None, None

    print("Running topic modeling (LDA)...")
    vectorizer = CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        stop_words="english"
    )
    doc_term_mat = vectorizer.fit_transform(documents)
    
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch"
    )
    lda_model.fit(doc_term_mat)
    print("Topic modeling completed.")
    return lda_model, vectorizer, doc_term_mat


def print_topics(lda_model, vectorizer, n_top_words: int = 10):
    """
    Prints the top words for each topic in an LDA model.
    """
    if lda_model is None or vectorizer is None:
        print("No topic model to display.")
        return
    
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"\nTopic #{topic_idx + 1}:")
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        print("  " + ", ".join(top_words))


def run_sentiment_analysis(documents: list, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Runs sentiment analysis on a list of text chunks using a Hugging Face model.
    
    Parameters:
        documents:  List of text chunks
        model_name: Name of the model to use for the pipeline
    
    Returns:
        List of sentiment analysis results, one per document.
    """
    if not documents:
        print("No documents provided for sentiment analysis.")
        return []
    
    print(f"Loading sentiment analysis model: {model_name} ...")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    print("Running sentiment analysis on chunks...")
    results = sentiment_pipeline(documents)
    print("Sentiment analysis completed.")
    return results


def summarize_sentiment(results: list):
    """
    Produces a simple summary of sentiment labels.
    
    Parameters:
        results: List of dictionaries with 'label' and 'score'
    """
    if not results:
        print("No sentiment results to summarize.")
        return
    
    from collections import Counter
    labels = [r["label"] for r in results]
    counts = Counter(labels)
    
    print("\nSentiment summary (by chunk):")
    total = len(labels)
    for label, count in counts.items():
        perc = 100 * count / total
        print(f"  {label}: {count} chunks ({perc:.1f}%)")

# ============================================================
# STEP 4: MAIN PIPELINE
# ============================================================

def main():
    # ----------------------------------------
    # 4.1 Upload audio file
    # ----------------------------------------
    audio_path = upload_audio_file()
    
    # ----------------------------------------
    # 4.2 Load Whisper model and transcribe
    # ----------------------------------------
    # You can change "base" to "small", "medium", "large" if you want.
    # For classroom recordings, "base" is a reasonable starting point.
    whisper_model = load_whisper_model(model_name="base")
    
    # If your lectures are always in English, set language="en" for stability.
    transcript = transcribe_audio(
        model=whisper_model,
        audio_path=audio_path,
        language="en"   # or None for auto-detect
    )
    
    # ----------------------------------------
    # 4.3 Save transcript to file
    # ----------------------------------------
    transcript_file = save_transcript(transcript, filename="lecture_transcript.txt")
    
    # Print first 500 characters as a preview
    print("\n=== TRANSCRIPT PREVIEW (first 500 characters) ===")
    print(transcript[:500])
    
    # ----------------------------------------
    # 4.4 Prepare text chunks for topic modeling & sentiment
    # ----------------------------------------
    chunks = chunk_text(transcript, chunk_size=120)
    print(f"\nTranscript split into {len(chunks)} chunks for analysis.")
    
    # ----------------------------------------
    # 4.5 Topic Modeling
    # ----------------------------------------
    lda_model, vectorizer, doc_term_mat = run_topic_modeling(
        documents=chunks,
        n_topics=5,          # number of topics you want to see
        max_features=1000
    )
    
    print("\n=== DISCOVERED TOPICS ===")
    print_topics(lda_model, vectorizer, n_top_words=10)
    
    # ----------------------------------------
    # 4.6 Sentiment Analysis
    # ----------------------------------------
    sentiment_results = run_sentiment_analysis(
        documents=chunks,
        model_name="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # Show first few results
    print("\n=== SAMPLE SENTIMENT RESULTS (first 5 chunks) ===")
    for i, res in enumerate(sentiment_results[:5]):
        print(f"Chunk {i+1}: label={res['label']}, score={res['score']:.3f}")
    
    # Summarize sentiment
    summarize_sentiment(sentiment_results)
    
    print("\nPipeline finished.")


# ============================================================
# STEP 5: RUN MAIN (ONLY IN INTERACTIVE ENVIRONMENTS)
# ============================================================
# In Colab, this will run the whole pipeline when you execute the cell.

if __name__ == "__main__":
    main()
