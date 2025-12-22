# Tourism Guide RAG System - Gemini

A RAG (Retrieval-Augmented Generation) system for tourism content using Google's Gemini AI.

## Overview

This system allows you to upload tourism content (Word docs, PDFs, text files) and create an interactive Q&A chatbot that answers questions based only on the uploaded content.

## File Structure

### Core Scripts

- **`main_upload.py`** - Upload and chunk tourism content
- **`main_interactive.py`** - Interactive Q&A chat interface

### Supporting Modules

- **`config.py`** - Configuration management
- **`chunker.py`** - Text chunking
- **`file_parser.py`** - Parse file formats
- **`store_manager.py`** - Gemini store management
- **`store_registry.py`** - Track stores
- **`query_logger.py`** - Log all queries
- **`utils.py`** - Utility functions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### 3. Configure API Key

Add your API key to `~/.bashrc`:

```bash
nano ~/.bashrc
```

Add this line at the end:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

Save and reload:
```bash
source ~/.bashrc
```

Verify it's set:
```bash
echo $GEMINI_API_KEY
```

### 4. Upload Content

```bash
python gemini/main_upload.py
```

### 5. Start Chatting

```bash
python gemini/main_interactive.py
```

## Query Logs

All queries logged to `gemini/query_log.jsonl` with timestamps, questions, answers, response times, and context.
