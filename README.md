# SentimentScope AI

An Intelligent Sentiment Analysis Platform Comparing Traditional Machine Learning and Transformer-Based NLP Models.

---

## 📌 Project Overview

**SentimentScope AI** is an end-to-end Natural Language Processing (NLP) application designed to analyze and compare sentiment classification performance across different types of text.

The system compares:

- Traditional TF-IDF-based Machine Learning models
- Transformer-based DistilBERT
- Performance across long-form movie reviews and informal social-media text
- Model predictions and confidence scores through an interactive web application

The project was developed as a complete NLP pipeline, covering dataset validation, exploratory data analysis, preprocessing, feature engineering, model training, evaluation, transformer fine-tuning, REST API development, and frontend visualization.

---

## 🎯 Problem Statement

Sentiment classification can behave differently depending on the characteristics of the text.

Long-form movie reviews generally contain structured language and detailed opinions, while social-media text can contain:

- Slang
- Short expressions
- Informal language
- Context-dependent sentiment
- Noisy text

This project investigates how traditional NLP models compare with Transformer-based models across these different domains.

---

# 📊 Datasets

## 1. IMDb Movie Reviews

- Total dataset: **50,000 reviews**
- Sentiment classes:
  - Positive
  - Negative
- Dataset type: Long-form movie reviews
- Train/Test structure used by the project:
  - Training data used for traditional ML: **40,000 samples**
  - Official test set: **5,000 samples**
  - Validation data used during experimentation: **5,000 samples**

The IMDb dataset represents structured, long-form opinionated text.

---

## 2. Twitter Sentiment140

- Project dataset: **50,000 samples**
- Sentiment classes:
  - Positive
  - Negative
- Dataset type: Informal social-media text
- Traditional ML training: **40,000 samples**
- Official test set: **5,000 samples**
- Validation data: **5,000 samples**

The Twitter dataset represents short, informal and context-dependent text.

---

# 🧠 Models

The project evaluates three traditional/standard ML approaches and one Transformer architecture.

### Traditional Machine Learning

1. **Naive Bayes**
2. **Logistic Regression**
3. **Linear SVM**

These models use **TF-IDF** features extracted from preprocessed text.

### Transformer Model

4. **DistilBERT**

Model:

```text
distilbert-base-uncased
DistilBERT was fine-tuned using the Hugging Face Transformers framework.

### 🔬 Project Phases
#### Phase 1 — Project Initialization & Setup

Completed.

Activities included:

- Project structure creation
- Python environment setup
- Dependency configuration
- NLP project architecture
- Testing infrastructure

#### Phase 2 — Dataset Acquisition, Validation & EDA

Completed.

Activities included:

- Dataset loading
- Dataset validation
- Missing-value checking
- Duplicate checking
- Class-distribution analysis
- Text-length analysis
- Dataset statistics
- Exploratory visualizations

#### Phase 3 — NLP Preprocessing & Traditional ML

Completed.

Pipeline:

Raw Text
   ↓
Text Cleaning
   ↓
Tokenization / Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Evaluation

Traditional models were trained using the available 40,000 training samples.

#### Phase 4 — Transformer-Based Sentiment Analysis

Completed.

DistilBERT was introduced to investigate contextual NLP performance.

Pilot Experiment

A 500-sample subset was initially used to validate:

- Transformer installation
- Tokenization
- Training pipeline
- Inference
- Evaluation
- CPU compatibility
- Fair Evaluation

A larger 5,000-sample stratified training subset was subsequently used for a fairer comparison.

For each dataset:

Training samples: 5,000
Validation samples: 1,000
Official test samples: 5,000
Training epochs: 2

The official test sets remained completely untouched during training and validation.

📈 Model Performance
IMDb
Model	F1-Score	Accuracy
Logistic Regression	90.72%	90.72%
Linear SVM	90.52%	90.52%
Naive Bayes	88.43%	88.44%
DistilBERT — Fair 5k	85.36%	85.36%
Observation

Traditional TF-IDF-based models performed extremely well on IMDb.

Logistic Regression achieved the highest performance with approximately 90.7% F1-score.

DistilBERT achieved 85.36% F1-score while being trained on only 5,000 samples compared with 40,000 training samples used by the traditional models.

Twitter Sentiment140
Model	F1-Score	Accuracy
DistilBERT — Fair 5k	80.34%	80.34%
Logistic Regression	77.32%	77.32%
Naive Bayes	76.35%	76.36%
Linear SVM	75.56%	75.56%
Observation

DistilBERT outperformed all traditional models on the Twitter dataset.

It achieved:

DistilBERT       80.34%
Logistic Reg.    77.32%

Despite being trained on only 5,000 samples, compared with 40,000 samples for the traditional ML models.

This demonstrates the potential advantage of contextual Transformer representations when dealing with informal social-media language.

⚖️ Scientific Fairness

The project intentionally separates:

Traditional ML
40,000 training samples
        ↓
TF-IDF
        ↓
Traditional ML
DistilBERT
5,000 training samples
        ↓
Tokenizer
        ↓
DistilBERT Fine-tuning

The 5,000 samples refer to training data, not test data.

The official test sets were kept separate and were not used for:

Training
Hyperparameter tuning
Model fitting

Final metrics were calculated using the official held-out test sets.

🏗️ System Architecture
                 ┌──────────────────────┐
                 │      User Input      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    React Frontend    │
                 │      Vite + UI       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     FastAPI API      │
                 └──────────┬───────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
      Logistic Regression  Linear SVM   DistilBERT
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Sentiment Prediction │
                 │ + Confidence Score   │
                 └──────────────────────┘
🌐 Web Application

The project includes a React-based frontend with four main pages.

1. Analyze

Allows the user to:

Select dataset/domain
Select a model
Enter custom text
Generate sentiment prediction
View confidence score
2. Compare

Runs the same text through all three integrated models:

Logistic Regression
Linear SVM
DistilBERT

The interface displays:

Predicted sentiment
Confidence score
Model agreement

Example:

Logistic Regression → POSITIVE → 98.6%
Linear SVM          → POSITIVE → 93.0%
DistilBERT          → POSITIVE → 96.4%

Model Agreement: 3 / 3
3. Performance Dashboard

The dashboard visualizes verified model evaluation results using Recharts.

It includes:

Model comparison
IMDb performance
Twitter performance
Traditional ML results
DistilBERT results
Training-data scaling experiment
4. About

Provides information about:

Project objectives
Datasets
NLP methodology
Models
Hardware limitations
Experimental methodology
🔌 Backend API

The backend is implemented using FastAPI.

Available Endpoints
Method	Endpoint	Description
GET	/api/health	Check API status
GET	/api/models	Get available models/domains
GET	/api/metrics	Get verified evaluation metrics
POST	/api/predict	Predict sentiment
POST	/api/compare	Compare all integrated models

Interactive API documentation is automatically available through FastAPI.

http://127.0.0.1:8000/docs
🧪 Testing

The project includes automated tests covering:

Data loading
Dataset validation
Preprocessing
TF-IDF
Traditional ML models
DistilBERT
Evaluation
API integration
Final Test Status
21 tests
21 passed
0 failed

The API integration tests specifically verify:

Logistic Regression prediction
Linear SVM prediction
DistilBERT prediction
Positive predictions
Negative predictions
Compare endpoint
🛠️ Technology Stack
Backend
Python
FastAPI
Uvicorn
scikit-learn
pandas
NumPy
PyTorch
Hugging Face Transformers
NLTK
Frontend
React
Vite
Recharts
Lucide React
CSS
Glassmorphism UI
Machine Learning
TF-IDF
Naive Bayes
Logistic Regression
Linear SVM
DistilBERT
Development
Git
GitHub
VS Code
Python Virtual Environment
npm
📁 Project Structure
SentimentSenseAI/
│
├── backend/
│   ├── api/
│   │   ├── routes_metrics.py
│   │   └── routes_predict.py
│   ├── schemas/
│   │   └── schemas.py
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── DATASET_SETUP.md
│   ├── DATA_DICTIONARY.md
│   └── MODEL_ARCHITECTURE.md
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_traditional_ml.ipynb
│   └── 04_transformer.ipynb
│
├── reports/
│   ├── figures/
│   ├── FINAL_TEST_EVALUATION.md
│   ├── model_results.csv
│   └── transformer_comparison.md
│
├── scripts/
│   ├── prepare_datasets.py
│   ├── run_eda.py
│   ├── train_traditional_models.py
│   ├── train_distilbert.py
│   └── train_distilbert_fair.py
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── preprocessing/
│
├── tests/
│   ├── test_api_integration.py
│   ├── test_data_loading.py
│   ├── test_data_validation.py
│   ├── test_distilbert.py
│   ├── test_evaluation.py
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_tfidf.py
│
├── .env.example
├── .gitignore
├── PROJECT_INSTRUCTIONS.md
├── README.md
└── requirements.txt
🚀 Installation
1. Clone the Repository
git clone https://github.com/JohenPerera123/SentimentSenseAI.git
cd SentimentSenseAI
2. Create Python Virtual Environment
Windows
python -m venv venv

Activate the environment:

venv\Scripts\activate

If PowerShell execution policy prevents activation, you can directly use:

.\venv\Scripts\python.exe
3. Install Python Dependencies
.\venv\Scripts\python.exe -m pip install -r requirements.txt
🌐 Frontend Setup

Open another terminal:

cd frontend
npm install

The project requires a compatible modern Node.js version.

Check:

node -v
npm.cmd -v
▶️ Running the Application

Two terminals are required.

Terminal 1 — Backend

From the project root:

.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs
Terminal 2 — Frontend
cd frontend
npm run dev

Frontend:

http://localhost:5173
⚠️ Hardware Limitation

Transformer fine-tuning is computationally expensive on CPU.

The local development environment used CPU-based training, therefore the project intentionally used a 5,000-sample stratified training subset for the fair DistilBERT evaluation.

Training time for the fair evaluation was approximately:

IMDb:    ~1 hour 21 minutes
Twitter: ~1 hour 12 minutes

Total:   ~2 hours 33 minutes

The implementation is designed so that the Transformer pipeline can be scaled to larger datasets using GPU infrastructure.

🔐 Data & Model Files

Large raw datasets, virtual environments, trained model checkpoints and generated binary model files are excluded from GitHub using .gitignore.

Examples:

venv/
data/raw/
data/processed/
models/
*.pkl
*.joblib
*.safetensors
*.pt
*.pth

This keeps the GitHub repository lightweight and reproducible.

Dataset setup instructions are available in:

docs/DATASET_SETUP.md
🔍 Key Findings
IMDb

Traditional TF-IDF-based models performed extremely well.

Logistic Regression achieved approximately 90.72% F1-score, making it the strongest model for the IMDb experiment.

Twitter

DistilBERT achieved the highest performance:

DistilBERT       80.34%
Logistic Reg.    77.32%
Naive Bayes      76.35%
Linear SVM       75.56%

This demonstrates the potential benefit of contextual Transformer representations for informal text.

⚠️ Limitations
DistilBERT was trained using a 5,000-sample subset rather than the complete training dataset.
Transformer training was performed on CPU due to local hardware limitations.
Traditional ML models and DistilBERT therefore do not have identical training-data sizes.
The results should not be interpreted as proof that Transformers always outperform traditional ML.
The experiment demonstrates model behavior under the specific dataset, training-size and hardware conditions used in this project.
🔮 Future Improvements

Possible future extensions include:

GPU-based DistilBERT training
Full-scale Transformer training
Hyperparameter optimization
Additional Transformer architectures
Multiclass sentiment classification
Explainable AI techniques
More social-media datasets
Cloud deployment
Docker containerization
Model monitoring
Real-time sentiment analytics
👨‍💻 Project Status

Status: Completed ✅

The project currently includes:

✅ Dataset validation
✅ EDA
✅ NLP preprocessing
✅ TF-IDF feature engineering
✅ Traditional ML models
✅ DistilBERT fine-tuning
✅ Fair Transformer evaluation
✅ Model comparison
✅ FastAPI backend
✅ React frontend
✅ Performance dashboard
✅ API integration tests
✅ 21/21 tests passing
✅ GitHub repository
📜 License

This project was developed for academic and educational purposes.