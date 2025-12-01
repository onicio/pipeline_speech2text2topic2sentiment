# Lecture Audio → Transcript, Topics & Sentiment (Whisper + Colab)

This repo contains a Google Colab–friendly Python script that turns recorded classes (or any lecture-style audio) into structured, analyzable text. The pipeline goes from **audio file** to **Whisper transcription**, **topic modeling**, and **sentiment analysis** in a single run.

## What this does

1. **Upload audio**
   - Uses Colab’s `files.upload()` to upload a local audio file  
   - Works with common formats like `.mp3`, `.m4a`, `.wav`, etc.

2. **Transcribe with Whisper**
   - Uses the open-source `openai-whisper` library (local model, no API key)
   - Default model: `base` (you can change to `tiny`, `small`, `medium`, `large`)
   - Designed with recorded classes in mind (lectures, talks, seminars)

3. **Save transcript**
   - Generates a `lecture_transcript.txt` file
   - Prints a short preview of the transcript in the notebook

4. **Topic modeling (LDA)**
   - Splits the transcript into word-based chunks
   - Uses `CountVectorizer` + `LatentDirichletAllocation` from scikit-learn
   - Prints top words for each discovered topic to help you see what the class focused on

5. **Sentiment analysis**
   - Uses Hugging Face Transformers sentiment pipeline
   - Runs sentiment analysis on each chunk of the transcript
   - Outputs per-chunk sentiment and a simple summary (how many chunks are positive/negative, etc.)

## Tech stack

- **Speech-to-text:** [`openai-whisper`](https://github.com/openai/whisper)
- **ML / NLP:**
  - `transformers` (Hugging Face sentiment analysis)
  - `scikit-learn` (CountVectorizer, LDA)
- **Environment:** Google Colab (CPU or GPU)

## How to use (in Google Colab)

1. Open a new notebook in [Google Colab](https://colab.research.google.com/).
2. Copy the script from this repo into a cell (or import it as a module).
3. Run the cell.
4. When prompted, upload your audio file (e.g., `lecture1.m4a`).
5. Wait for:
   - Whisper transcription  
   - Topic modeling  
   - Sentiment analysis  
6. Download `lecture_transcript.txt` from the Colab file browser if needed.

## Adapting the pipeline

- Change the Whisper model:
  ```python
  whisper_model = load_whisper_model(model_name="small")
