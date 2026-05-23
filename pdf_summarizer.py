# *** COPYRIGHT NOTICE ***
# © 2025 Ozgur Eralp. All Rights Reserved.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
from gtts import gTTS
import os
import asyncio
import base64
import threading
import torch

# Model selection and explicit revision
model_name = "sshleifer/distilbart-cnn-12-6"
model_revision = "a4f8f3e"

# Hardware acceleration detection for MacBook (MPS) or other platforms
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# model_name = "facebook/bart-large-cnn"  # Or any other model name
# model_revision = "a4f8f3e" # Or any other model revision, or delete it to use the latest one.

class RobustSummarizer:
    """A fallback summarizer that bypasses the transformers pipeline registry and class imports."""
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, text, max_length=100, min_length=25, do_sample=False):
        # BART models typically have a maximum input length of 1024 tokens
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=do_sample
        )
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return [{"summary_text": summary_text}]

# Global variable to store the summarizer pipeline instance (lazy loading)
_summarizer = None
_summarizer_lock = threading.Lock()

def get_summarizer():
    """Initializes and returns the summarizer pipeline lazily with thread safety."""
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    with _summarizer_lock:
        if _summarizer is None:
            print(f"Initializing summarizer with model: {model_name}...")
            try:
                # Explicitly load tokenizer and model to handle environment inconsistencies
                tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=model_revision)

                # Bypass the task registry entirely using direct model usage
                _summarizer = RobustSummarizer(model, tokenizer, device)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize the summarization pipeline: {e}")
    return _summarizer

async def display_pdf_preview(file_obj):
    if file_obj is None:
        return ""
    try:
        with open(file_obj, "rb") as pdf_file:
            pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')

        pdf_embed = f'<embed id="pdf-embed" src="data:application/pdf;base64,{pdf_base64}" width="100%" height="500px" type="application/pdf" style="transform: scale(1); transform-origin: top left;">'

        return pdf_embed
    except Exception as e:
        return f"Error generating PDF preview: {e}"

async def summarize_pdf(file_obj, max_chunk_length, max_summary_length):
    print(f"summarize_pdf working on {file_obj}")
    min_summary_length = max_summary_length // 2
    
    try:
        if file_obj is None:
            return "Please upload a PDF file.", None

        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        if not text.strip():
            return "The uploaded PDF appears to be empty or contains no text.", None

        words = text.split()
        text_chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > max_chunk_length:
                text_chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        if current_chunk:
            text_chunks.append(" ".join(current_chunk))

        summarized_text = ""
        # Get the summarizer instance only when needed
        summarizer_pipeline = get_summarizer()
        for chunk in text_chunks:
            if len(chunk) >= min_summary_length:
                summary = summarizer_pipeline(chunk, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)
                summarized_text += summary[0]['summary_text']

        tts = gTTS(text=summarized_text)
        pdf_filename = os.path.splitext(os.path.basename(file_obj))[0]
        
        # Ensure the output directory exists
        if not os.path.exists("./mp3"):
            os.makedirs("./mp3")
            
        audio_path = f"./mp3/{pdf_filename}_summary.mp3"
        await asyncio.to_thread(tts.save, audio_path)

        print(f"summarize_pdf done {audio_path}")
        return summarized_text, audio_path

    except (FileNotFoundError, PyPDF2.errors.PdfReadError) as e: # Specifically handle file errors
        return f"Error reading PDF: {e}", None
    except Exception as e:
        return f"An error occurred: {e}", None
