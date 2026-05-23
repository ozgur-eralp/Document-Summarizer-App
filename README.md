# Analysis App: PDF Summarizer (NLP & TTS)(POC)

📢 **Public Release Note**: This project is an experimental trial that I am opening up to the public. It is shared "as-is" in the hope that it might spark inspiration, serve as a playground and helpful reference, or act as a building block for someone else working at the intersection of document analysis and web-based AI applications. It was originally developed in February 2025 as a **POC - Proof of Concept**.

## 📄 Overview

Analysis App is a web tool designed for document analysis and accessibility. It utilizes state-of-the-art Natural Language Processing (NLP) to extract text from PDF files, generate concise, news-style summaries using the `distilbart-cnn-12-6` transformer model, and automatically produce audio reports via Google Text-to-Speech (gTTS).

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (Tested up to experimental 3.14)
- MacOS (Silicon/Intel), Linux, or Windows

### 2. Setup Environment
It is highly recommended to use a virtual environment to manage dependencies:

```bash
# Create the environment
python3 -m venv .venv 

# Activate on macOS/Linux
source .venv/bin/activate 

# Activate on Windows
# .venv\Scripts\activate
```

### 3. Installation
Install the core libraries and the PyTorch framework (required for the inference engine):

```bash
pip install gradio PyPDF2 transformers gTTS torch
```

## 🛠 Running the Application

Launch the application by executing the main script:

```bash
python3 app.py
```

**Note on First Run:**
The system will automatically download the `distilbart-cnn-12-6` model (approx. 1.2GB) upon startup. The UI will display a "System Initializing" status until the weights are fully loaded into memory and hardware acceleration is configured.

## 🧠 Technical Details

### Hardware Acceleration
The application automatically detects the best available backend for your machine:
- **Apple Silicon (MPS):** Utilizes Metal Performance Shaders for high-speed inference on MacBooks.
- **NVIDIA (CUDA):** Utilizes GPU acceleration on supported Windows/Linux systems.
- **CPU:** Standard fallback if no supported accelerator is found.

### Robust Inference Engine
To ensure stability across various Python environments (including experimental versions), the app implements a `RobustSummarizer` class. This class bypasses the standard transformers pipeline registry and handles tokenization and tensor placement directly, ensuring consistent behavior regardless of environment quirks.

### Model: DistilBART-CNN-12-6
- **Efficiency:** A "distilled" version of BART, offering a significant reduction in parameters and latency while maintaining high summary quality.
- **CNN Optimized:** Specifically fine-tuned for news and report-style text structures.
- **Architecture:** 12 encoder layers and 6 decoder layers.

## 📂 Project Structure
- `app.py`: UI Layout, state management, and event handling.
- `pdf_summarizer.py`: PDF text extraction, AI inference logic, and TTS generation.
- `mp3/`: Local directory generated to store audio summary outputs.

## 📜 Credits & License

Developed by **Ozgur Eralp** (All rights reserved 2025).  
Inference: Hugging Face Transformers (Python)
