# SentimentScope API & Application

An Intelligent Sentiment Analysis Platform Using Traditional Machine Learning and Transformer-Based Language Models.

## Project Overview
This project is an NLP application that performs sentiment classification using traditional TF-IDF-based machine learning models (Naive Bayes, Logistic Regression, SVM) and transformer-based language models (DistilBERT).

### Current Status
- **Phase 1**: Project Initialization & Setup (Completed)
- **Phase 2**: Dataset Acquisition, Validation & EDA (Completed)
- **Phase 3**: NLP Preprocessing, TF-IDF & Traditional ML (Completed)
- **Phase 4**: Transformer-Based Sentiment Analysis with DistilBERT (Completed)
- **Phase 5**: Web API & Frontend (Completed)

## Problem Statement
Analyzing sentiment on movie reviews vs informal social media text.

## Datasets
- **IMDb Movie Reviews**: 50,000 structured movie reviews, labeled positive/negative.
- **Twitter Sentiment140**: 50,000 sample of informal social media tweets, labeled positive/negative.

## Technology Stack
- **Backend**: Python, FastAPI, Uvicorn, scikit-learn, PyTorch, Hugging Face Transformers.
- **Frontend**: React, Vite, Recharts, Lucide React (Vanilla modern CSS with glassmorphism).

## Features
- **Analyze Page**: Select a domain, choose an NLP model (Logistic Regression, SVM, DistilBERT), and predict sentiment for custom text.
- **Compare Page**: Enter a single text and see the predictions and confidence scores of all 3 models side-by-side. 
- **Performance Dashboard**: Verified Phase 3.5 and 4.5 test metrics rendered dynamically using Recharts.

## Installation & Setup

### 1. Backend Environment
```bash
# Assuming you have Python installed, create a virtual environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend Environment
```bash
cd frontend
npm install
```

## Running Locally (Windows)

You need to run two terminal processes.

### Terminal 1: FastAPI Backend
```bash
venv\Scripts\activate
python -m uvicorn backend.main:app --reload --port 8000
```
*API will be available at http://localhost:8000/api*

### Terminal 2: React Frontend
```bash
cd frontend
npm run dev
```
*Application will be available at http://localhost:5173*

## API Endpoints
- `GET /api/health`: Check API status.
- `GET /api/models`: Return available domains and models.
- `GET /api/metrics`: Return static, verified test-set metrics.
- `POST /api/predict`: Predict sentiment for a specific model/domain.
- `POST /api/compare`: Run predictions across all models for a domain.

## Key Findings & Limitations
1. **Traditional ML is highly competitive**: On structured long-form text (IMDb), Logistic Regression achieved ~90.7% F1-score.
2. **Transformers rule informal context**: On Twitter data, DistilBERT (trained on 5,000 samples) achieved ~80.3% F1, outperforming Logistic Regression's ~77.3% (trained on 40,000 samples). Contextual NLP is critical for slang and informal semantics.
3. **Hardware Constraint**: Training transformers is massively compute-heavy. We utilized a 5,000-sample fair subset for DistilBERT on CPU to prevent excessive train times, proving the logic works and is ready to scale to GPUs for the full 40,000 sample dataset.
