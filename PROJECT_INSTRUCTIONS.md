# SentimentSense AI

## Complete NLP Mini Project — Master Development Instructions

---

## 1. ROLE

You are the lead software engineer, NLP engineer, machine learning engineer, UI/UX engineer, tester, and technical documentation assistant responsible for developing this entire project.

Build the project as a **real, working, portfolio-quality NLP application**, not as a toy/demo with fake outputs.

The final application must run locally on Windows and provide an interactive web interface for sentiment analysis.

The project must be understandable enough for a university NLP mini-project presentation and viva.

---

# 2. PROJECT TITLE

## SentimentSense AI

### An Intelligent Sentiment Analysis Platform Using Traditional Machine Learning and Transformer-Based Language Models

---

# 3. ACADEMIC CONTEXT

This is an NLP mini-project.

The project must demonstrate the complete NLP workflow:

1. Text data collection
2. Data exploration
3. Text cleaning
4. Tokenization
5. Stopword handling
6. Stemming/Lemmatization where appropriate
7. Text representation
8. TF-IDF
9. Traditional machine learning classification
10. Transformer-based language model
11. Model comparison
12. Evaluation
13. Error analysis
14. Interactive prediction
15. Web application

The project must explicitly demonstrate the concepts covered in the provided NLP lecture material.

Important lecture concepts include:

* NLP introduction
* Social media and review text
* Text preprocessing
* Word and sub-word tokenization
* Bag of Words
* TF-IDF
* Word embeddings
* Contextual embeddings
* Transformer architecture
* BERT
* Traditional ML baselines
* Transformer-based sentiment classification
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Sarcasm
* Irony
* Mixed sentiment
* Domain-specific language
* Multilingual/code-switching as possible future work

Do not remove these concepts from the project merely to simplify implementation.

---

# 4. MAIN RESEARCH QUESTION

The project should investigate:

> How effectively do traditional TF-IDF-based machine learning models and transformer-based language models perform for sentiment classification across movie-review and social-media text?

Additional research questions:

1. Which traditional ML model performs best on sentiment classification?
2. Does a transformer model outperform traditional TF-IDF-based models?
3. How does a model trained on one text domain perform on another domain?
4. How do informal social-media characteristics affect sentiment classification?
5. What types of texts are difficult for the models?

---

# 5. DATASETS

Use two datasets.

## Dataset 1 — IMDb Movie Reviews

Purpose:

* Main supervised sentiment dataset
* Training and evaluation
* Traditional ML experiments
* Transformer experiments

Expected characteristics:

* Approximately 50,000 reviews
* Positive and negative sentiment
* Movie-review domain

Expected file:

```text
IMDB Dataset.csv
```

Expected columns:

```text
review
sentiment
```

Store at:

```text
data/raw/imdb/IMDB Dataset.csv
```

---

## Dataset 2 — Sentiment140

Purpose:

* Social-media sentiment analysis
* Domain comparison
* Cross-domain/generalization experiment
* Testing informal language handling

Expected characteristics:

* Large Twitter dataset
* Approximately 1.6 million tweets
* Binary sentiment labels

Do NOT load the entire dataset into memory unnecessarily.

Create a reproducible sample of approximately:

```text
50,000 tweets
```

or another computationally reasonable size.

Store raw data at:

```text
data/raw/twitter/
```

After preprocessing/sampling, create:

```text
data/processed/twitter_sample.csv
```

---

# 6. DATASET RULES

Do not fabricate datasets.

Do not fabricate labels.

Do not fabricate model results.

Do not hard-code accuracy values.

All reported metrics must come from actual experiments.

If a dataset is missing, clearly report that the dataset is unavailable and provide the exact expected file location.

Do not silently create fake sample data and present it as real data.

---

# 7. IMPORTANT DATA SPLIT STRATEGY

Use reproducible train/validation/test splits.

For the IMDb dataset:

```text
80% training
10% validation
10% testing
```

For traditional ML models, use:

```text
training set → fit model
validation set → optional tuning
test set → final evaluation
```

For transformer training:

```text
80% training
10% validation
10% testing
```

Use fixed random seeds.

Example:

```python
RANDOM_STATE = 42
```

Document all split strategies.

---

# 8. EXPERIMENT DESIGN

Run at least the following experiments.

## Experiment A — Traditional ML on IMDb

Use:

1. Naive Bayes
2. Logistic Regression
3. Linear SVM

Feature representation:

```text
TF-IDF
```

Compare:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## Experiment B — Transformer on IMDb

Use a lightweight transformer model suitable for local development.

Preferred model:

```text
DistilBERT
```

Possible Hugging Face model:

```text
distilbert-base-uncased
```

If local hardware is insufficient for fine-tuning, implement the transformer inference architecture correctly and document the limitation rather than fabricating training results.

Do not claim BERT/DistilBERT accuracy without actually running evaluation.

---

## Experiment C — Traditional ML on Twitter

Use:

* TF-IDF
* Logistic Regression
* Linear SVM
* Naive Bayes where computationally reasonable

Evaluate using the same metrics.

---

## Experiment D — Cross-Domain Evaluation

Perform at least one meaningful cross-domain experiment.

Example:

```text
Train on IMDb
        ↓
Traditional ML / Transformer
        ↓
Evaluate on Twitter sample
```

If label semantics are incompatible, document the limitation instead of forcing an invalid comparison.

Also consider:

```text
Train on Twitter sample
        ↓
Evaluate on IMDb
```

only if the label mapping is valid.

---

# 9. TEXT PREPROCESSING

Create a reusable preprocessing pipeline.

The pipeline should handle:

1. Lowercasing
2. HTML tag removal
3. URL removal
4. Mention removal
5. Hashtag handling
6. Punctuation handling
7. Number handling where appropriate
8. Extra whitespace
9. Emoji handling
10. Stopword handling
11. Tokenization
12. Lemmatization or stemming where appropriate

Important:

Do NOT blindly remove information that can affect sentiment.

For example:

```text
not good
```

must not become:

```text
good
```

because negation is sentiment-critical.

Design preprocessing separately where necessary for:

```text
traditional TF-IDF models
```

and:

```text
transformer models
```

Transformer models should preserve contextual information as much as possible.

---

# 10. SOCIAL MEDIA PREPROCESSING

Twitter text may contain:

* @mentions
* #hashtags
* URLs
* emojis
* abbreviations
* repeated characters
* slang
* informal spelling

Do not destroy these features unnecessarily.

Create sensible normalization.

For example:

```text
"LOVE this movie!!! 😍😍 https://example.com"
```

may become something like:

```text
"love this movie"
```

for a traditional TF-IDF pipeline.

But retain appropriate information for transformer experiments when possible.

Document these decisions.

---

# 11. TOKENIZATION

Demonstrate:

## Word Tokenization

For traditional NLP processing.

## Subword Tokenization

For transformer models.

Explain the difference in documentation.

---

# 12. FEATURE REPRESENTATION

Implement:

## TF-IDF

Use scikit-learn.

Document:

* vocabulary
* n-grams
* max features
* minimum document frequency
* maximum document frequency
* normalization

Use reasonable parameters rather than blindly using defaults.

Example starting point:

```python
TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

Tune only when computationally reasonable.

---

# 13. TRADITIONAL ML MODELS

Implement:

### Naive Bayes

Use:

```text
MultinomialNB
```

### Logistic Regression

Use:

```text
LogisticRegression
```

### Linear SVM

Use:

```text
LinearSVC
```

Use pipelines where appropriate.

Save trained models using appropriate serialization.

Never retrain models every time the web application starts.

---

# 14. TRANSFORMER MODEL

Use:

```text
DistilBERT
```

Preferred Hugging Face architecture:

```text
AutoTokenizer
AutoModelForSequenceClassification
```

Number of classes:

```text
2
```

Labels:

```text
0 = Negative
1 = Positive
```

Starting hyperparameters may be:

```text
learning rate = 2e-5
epochs = 2–3
batch size = 8 or 16 depending on hardware
max sequence length = 128 or 256
```

Do not assume these values are optimal.

Document actual values used.

Use early stopping or the best validation checkpoint when practical.

---

# 15. COMPUTATIONAL SAFETY

The developer environment may not have a dedicated GPU.

Therefore:

* Detect whether CUDA is available.
* Use GPU if available.
* Otherwise use CPU.
* Do not crash because CUDA is unavailable.
* Keep transformer training computationally reasonable.
* Make dataset size configurable.
* Make batch size configurable.
* Make maximum sequence length configurable.

Example configuration:

```text
USE_GPU_IF_AVAILABLE=true
MAX_LENGTH=128
BATCH_SIZE=8
EPOCHS=2
```

Do not require the user to train a huge model from scratch.

---

# 16. MODEL SAVING

After training, save models.

Example:

```text
models/
├── tfidf/
├── naive_bayes/
├── logistic_regression/
├── svm/
└── distilbert/
```

The web application must load saved models.

It must NOT retrain models during normal prediction.

---

# 17. MODEL EVALUATION

For every model calculate:

```text
Accuracy
Precision
Recall
F1-score
```

For binary classification provide:

```text
Positive class metrics
Negative class metrics
Macro average
Weighted average
```

Also generate:

```text
Confusion Matrix
Classification Report
```

---

# 18. MODEL COMPARISON

Generate a comparison table.

Example structure:

| Model               | Dataset | Accuracy | Precision | Recall |     F1 |
| ------------------- | ------- | -------: | --------: | -----: | -----: |
| Naive Bayes         | IMDb    |   actual |    actual | actual | actual |
| Logistic Regression | IMDb    |   actual |    actual | actual | actual |
| SVM                 | IMDb    |   actual |    actual | actual | actual |
| DistilBERT          | IMDb    |   actual |    actual | actual | actual |
| Naive Bayes         | Twitter |   actual |    actual | actual | actual |
| Logistic Regression | Twitter |   actual |    actual | actual | actual |
| SVM                 | Twitter |   actual |    actual | actual | actual |
| DistilBERT          | Twitter |   actual |    actual | actual | actual |

IMPORTANT:

Never populate this table with invented numbers.

---

# 19. ERROR ANALYSIS

Implement an error-analysis module.

Identify examples of:

### Sarcasm

Example:

```text
"Great! Another software crash."
```

### Irony

### Mixed Sentiment

Example:

```text
"The movie was good but the ending was terrible."
```

### Domain-specific language

### Slang

### Negation

### Very short text

### Ambiguous text

Save incorrectly classified examples.

Create:

```text
reports/error_analysis.csv
```

Columns:

```text
text
actual_label
predicted_label
model
confidence
error_type
```

If automatic error-type classification is unreliable, allow manual categorization during analysis.

---

# 20. EXPLAINABILITY

The application should provide a simple explanation of predictions.

For traditional TF-IDF models:

Use influential TF-IDF features where technically appropriate.

For example:

```text
Important positive terms:
excellent
amazing
love
```

For negative:

```text
terrible
waste
boring
```

For transformer models, do not falsely claim that attention weights are automatically a complete explanation.

If implementing explainability, use an appropriate technique such as:

* token attribution
* SHAP where feasible
* Integrated Gradients where feasible

If explainability is too computationally expensive, provide a clearly labelled simplified explanation rather than pretending it is a faithful causal explanation.

---

# 21. EMOTION DETECTION

This is an optional enhancement.

If implemented, clearly separate:

```text
Sentiment
```

from:

```text
Emotion
```

Sentiment:

```text
Positive / Negative
```

Emotion:

```text
Joy
Anger
Sadness
Fear
Love
Disgust
```

Do not claim that the sentiment classifier itself performs emotion classification unless it was trained for that task.

If there is insufficient time, leave emotion detection as a future enhancement.

---

# 22. WEB APPLICATION

Build a complete local web application.

Architecture:

```text
React Frontend
       ↓
Flask REST API
       ↓
NLP Prediction Service
       ↓
Saved ML / Transformer Models
       ↓
SQLite Database
```

---

# 23. BACKEND

Use:

```text
Python
Flask
Flask-CORS
SQLite
SQLAlchemy if useful
```

Recommended structure:

```text
backend/
├── app.py
├── config.py
├── requirements.txt
│
├── api/
│   ├── routes.py
│   ├── prediction_routes.py
│   └── analytics_routes.py
│
├── services/
│   ├── prediction_service.py
│   ├── preprocessing_service.py
│   ├── model_service.py
│   └── analytics_service.py
│
├── models/
│   └── database_models.py
│
├── utils/
│   └── helpers.py
│
└── database/
    └── database.db
```

Keep business logic out of Flask route functions whenever possible.

---

# 24. API ENDPOINTS

Implement:

## Health

```http
GET /api/health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Prediction

```http
POST /api/predict
```

Input:

```json
{
  "text": "I absolutely love this movie!"
}
```

Output should contain actual model results, for example:

```json
{
  "sentiment": "Positive",
  "confidence": 0.97,
  "model": "DistilBERT"
}
```

Do not hard-code the result.

---

## Model Comparison

```http
POST /api/predict/compare
```

Return predictions from available models.

Example structure:

```json
{
  "text": "...",
  "models": [
    {
      "model": "Logistic Regression",
      "sentiment": "Positive",
      "confidence": 0.91
    },
    {
      "model": "SVM",
      "sentiment": "Positive"
    },
    {
      "model": "DistilBERT",
      "sentiment": "Positive",
      "confidence": 0.97
    }
  ]
}
```

---

## History

```http
GET /api/history
```

---

## Analytics

```http
GET /api/analytics
```

Return actual database-derived statistics.

---

# 25. DATABASE

Use SQLite.

Table:

```text
predictions
```

Fields:

```text
id
text
sentiment
confidence
model
dataset_domain if applicable
created_at
```

Do not store unnecessary personal information.

---

# 26. FRONTEND

Use:

```text
React
Vite
Tailwind CSS
Axios
Recharts
```

If Tailwind setup causes unnecessary configuration problems, use a clean CSS architecture instead of wasting project time.

The application must still look professional.

---

# 27. FRONTEND PAGES

Create:

## Home

Main prediction interface.

Components:

```text
Header
Text Input
Character Counter
Analyze Button
Loading State
Prediction Result
Confidence Score
Important Words
Model Information
```

---

## Model Comparison

Display:

```text
Naive Bayes
Logistic Regression
SVM
DistilBERT
```

side-by-side.

---

## Analytics Dashboard

Display:

* Total predictions
* Positive count
* Negative count
* Positive percentage
* Negative percentage
* Model usage
* Sentiment distribution
* Recent predictions
* Prediction trend

Use charts.

---

## History

Table:

```text
Text
Sentiment
Confidence
Model
Date
```

Add:

* search
* filtering
* pagination where useful

---

## About

Explain:

* project objective
* datasets
* NLP pipeline
* models
* technology stack
* limitations

---

# 28. UI DESIGN

The interface must look like a modern AI analytics application.

Requirements:

* responsive
* clean typography
* card-based layout
* clear sentiment indicators
* good spacing
* accessible buttons
* loading indicators
* error messages
* empty states
* mobile-friendly layout

Avoid excessive visual effects.

Prioritize usability.

---

# 29. HOME PAGE USER FLOW

User enters:

```text
"This movie was fantastic and I loved every minute of it."
```

Clicks:

```text
Analyze Sentiment
```

The application sends:

```text
POST /api/predict
```

Backend loads the selected model.

Backend performs prediction.

Backend returns:

```text
Sentiment
Confidence
Model
```

Frontend displays the result.

Prediction is saved to SQLite.

Dashboard statistics update.

---

# 30. MODEL SELECTION

Allow the user to select:

```text
Best Model
Logistic Regression
SVM
DistilBERT
```

If a selected model is unavailable because it has not been trained, show a clear message.

Never silently substitute a different model.

---

# 31. DEFAULT MODEL

Preferred default:

```text
DistilBERT
```

if successfully trained and saved.

Otherwise:

```text
Logistic Regression
```

as the reliable lightweight fallback.

---

# 32. DATA ANALYTICS

Create visualizations from actual data.

Required:

### Sentiment distribution

```text
Positive
Negative
```

### Model comparison

Bar chart:

```text
Accuracy
F1
```

### Confusion matrix

For each major model.

### Word frequency

Positive and negative vocabulary.

### Prediction history

Time-based trend.

Do not use fake chart data.

---

# 33. WORD CLOUD

Generate:

```text
Positive Word Cloud
Negative Word Cloud
```

Use actual dataset text.

Do not create artificial word clouds.

---

# 34. PROJECT CONFIGURATION

Create a configuration file.

Example:

```text
.env
```

Possible settings:

```text
FLASK_ENV=development
PORT=5000
DATABASE_URL=sqlite:///database.db
DEFAULT_MODEL=distilbert
MAX_TEXT_LENGTH=1000
```

Do not commit secrets.

Create:

```text
.env.example
```

---

# 35. ERROR HANDLING

The backend must handle:

* empty text
* extremely long text
* invalid request body
* model unavailable
* database error
* prediction error
* missing model file

Example:

```json
{
  "error": "Text cannot be empty."
}
```

Use proper HTTP status codes.

---

# 36. INPUT VALIDATION

Reject empty input.

Limit text length.

Normalize unsafe input.

Never execute user-provided text as code.

Do not expose server stack traces to frontend users.

---

# 37. TESTING

Create tests.

At minimum:

### Unit tests

Test:

* preprocessing
* model loading
* prediction service
* API validation

### API tests

Test:

```text
GET /api/health
POST /api/predict
GET /api/history
GET /api/analytics
```

### Frontend tests

Where practical:

* input rendering
* button interaction
* result rendering
* error state

---

# 38. ACCEPTANCE TEST

The final system must pass:

### Test 1

Input:

```text
I absolutely love this movie!
```

Expected general sentiment:

```text
Positive
```

The exact confidence must come from the actual model.

### Test 2

Input:

```text
This movie was terrible and boring.
```

Expected general sentiment:

```text
Negative
```

Again, use actual model output.

### Test 3

Empty input.

Expected:

```text
Validation error
```

### Test 4

Very long input.

Expected:

```text
Handled safely
```

### Test 5

API unavailable.

Expected:

```text
Friendly frontend error
```

---

# 39. LOGGING

Implement useful application logs.

Log:

* startup
* model loading
* API errors
* prediction failures

Do not log sensitive user data unnecessarily.

---

# 40. PERFORMANCE

Optimize for local development.

Do not load the transformer model repeatedly for every request.

Load it once when the service starts, if possible.

Use lazy loading only when necessary.

Cache model instances.

---

# 41. PROJECT STRUCTURE

Use this overall structure:

```text
SentimentSenseAI/
│
├── data/
│   ├── raw/
│   │   ├── imdb/
│   │   └── twitter/
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_traditional_ml.ipynb
│   ├── 04_transformer.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── models/
│   ├── tfidf/
│   ├── naive_bayes/
│   ├── logistic_regression/
│   ├── svm/
│   └── distilbert/
│
├── backend/
│
├── frontend/
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── error_analysis.csv
│
├── tests/
│
├── scripts/
│
├── docs/
│
├── .env.example
├── .gitignore
├── README.md
├── PROJECT_INSTRUCTIONS.md
└── requirements.txt
```

Adjust this structure if a better modular architecture is required, but preserve the separation of:

```text
data
NLP
models
backend
frontend
tests
documentation
```

---

# 42. PYTHON ENVIRONMENT

The target environment is Windows.

Expected environment:

```text
Python 3.13.1
Node.js 20.17.0
Git 2.47.1
```

Design the project accordingly.

Important:

If a dependency is incompatible with Python 3.13, identify it and use a compatible alternative/version.

Do not randomly downgrade the user's environment.

Prefer a virtual environment:

```text
venv/
```

Never commit:

```text
venv/
```

---

# 43. REQUIREMENTS

Generate:

```text
backend/requirements.txt
```

or a root requirements file.

Pin versions where appropriate after confirming compatibility.

Include only dependencies actually required.

Do not add unnecessary packages.

---

# 44. GIT

Initialize Git if not already initialized.

Create a useful `.gitignore`.

Ignore:

```text
venv/
__pycache__/
*.pyc
.env
database.db
large raw datasets
model checkpoints that are too large
node_modules/
dist/
build/
.ipynb_checkpoints/
```

Do NOT commit huge datasets or large model files unless explicitly required.

Provide instructions for downloading datasets and trained models.

---

# 45. README

Create a professional README containing:

## Project Overview

## Problem Statement

## Objectives

## Research Questions

## Datasets

## NLP Pipeline

## Model Architecture

## Traditional Models

## Transformer Model

## Experimental Setup

## Evaluation Metrics

## Results

## Error Analysis

## Web Application

## API Documentation

## Installation

## Running Backend

## Running Frontend

## Project Structure

## Limitations

## Future Work

## Authors

Do not invent author information.

Use placeholders if required.

---

# 46. DOCUMENTATION

Create documentation that explains the project in beginner-friendly language.

The developer should be able to explain:

* What is NLP?
* Why sentiment analysis?
* Why IMDb?
* Why Twitter?
* Why TF-IDF?
* Why Naive Bayes?
* Why Logistic Regression?
* Why SVM?
* Why DistilBERT?
* What is a Transformer?
* What is attention?
* Why does DistilBERT help?
* Why compare models?
* Why might Twitter be harder?
* What is cross-domain evaluation?
* What are the limitations?

---

# 47. ACADEMIC REPORT SUPPORT

Generate report-ready outputs.

Create:

```text
reports/
├── dataset_statistics.csv
├── model_results.csv
├── error_analysis.csv
└── figures/
```

Figures should include:

```text
class_distribution.png
model_accuracy_comparison.png
model_f1_comparison.png
confusion_matrix_*.png
positive_wordcloud.png
negative_wordcloud.png
```

Use actual experimental data.

---

# 48. PRESENTATION SUPPORT

Create a `docs/presentation_outline.md` containing approximately:

1. Title
2. Introduction
3. Problem Statement
4. Objectives
5. Datasets
6. NLP Pipeline
7. Preprocessing
8. TF-IDF
9. Traditional Models
10. Transformer / DistilBERT
11. Experimental Setup
12. Results
13. Model Comparison
14. Error Analysis
15. Web Application
16. Demo
17. Limitations
18. Future Work
19. Conclusion

Do not fabricate final numerical results.

---

# 49. FUTURE WORK

Mention potential future extensions:

* Multilingual NLP
* Sinhala-English code-switching
* Sinhala sentiment analysis
* Emotion detection
* Aspect-based sentiment analysis
* Sarcasm detection
* Larger transformer models
* RoBERTa
* multilingual transformers
* LLM-based zero-shot/few-shot classification

These are future enhancements unless actually implemented.

---

# 50. IMPORTANT: DO NOT OVERENGINEER

The project must be impressive but achievable.

Do not add unnecessary:

* microservices
* Docker
* Kubernetes
* cloud deployment
* authentication systems
* payment systems
* complex distributed databases

unless specifically requested later.

Focus on:

```text
NLP quality
ML quality
Transformer comparison
Evaluation
Web application
Clean architecture
Documentation
```

---

# 51. DEVELOPMENT PHASES

Build in this exact order.

## PHASE 1 — Project Setup

Create:

* folders
* virtual environment instructions
* Git
* configuration
* requirements
* README

Then STOP and verify.

---

## PHASE 2 — Dataset

Implement:

* dataset loading
* dataset validation
* dataset statistics
* duplicate detection
* missing values
* class distribution

Then STOP and verify.

---

## PHASE 3 — EDA

Generate:

* class distribution
* text length distribution
* word frequency
* word clouds
* dataset comparison

Then STOP and verify.

---

## PHASE 4 — Preprocessing

Implement reusable preprocessing.

Write tests.

Then STOP and verify.

---

## PHASE 5 — TF-IDF

Implement TF-IDF pipeline.

Save vectorizer.

Then STOP and verify.

---

## PHASE 6 — Traditional ML

Train:

* Naive Bayes
* Logistic Regression
* SVM

Evaluate.

Save models.

Then STOP and verify.

---

## PHASE 7 — Transformer

Implement DistilBERT.

Detect GPU/CPU.

Train only with configurable dataset size.

Evaluate.

Save model/tokenizer.

Then STOP and verify.

---

## PHASE 8 — Comparison

Generate:

* metrics table
* charts
* confusion matrices
* error analysis

Then STOP and verify.

---

## PHASE 9 — Flask API

Implement:

* health
* prediction
* model comparison
* history
* analytics

Test with Python requests or curl.

Then STOP and verify.

---

## PHASE 10 — React Frontend

Implement:

* Home
* Prediction
* Comparison
* Analytics
* History
* About

Then STOP and verify.

---

## PHASE 11 — Integration

Connect:

```text
React → Flask → Models → SQLite
```

Test end-to-end.

---

## PHASE 12 — Final QA

Run:

* backend tests
* API tests
* frontend tests
* prediction tests
* invalid input tests

Fix all critical errors.

---

# 52. AGENT BEHAVIOR RULES

You are an autonomous coding agent, but you must work carefully.

Before making major architectural changes:

1. Inspect the existing project.
2. Read relevant files.
3. Do not overwrite working code unnecessarily.
4. Explain what you intend to change.
5. Implement.
6. Test.
7. Report the result.

Do not continuously generate code without testing.

---

# 53. NO FAKE COMPLETION

Never say:

```text
Done
```

unless the requested task was actually implemented and tested.

Never claim:

```text
Model accuracy = 92%
```

unless the model actually produced that result.

Never claim:

```text
BERT is working
```

unless actual inference/training has been tested.

Never claim:

```text
API works
```

unless an actual API request has succeeded.

Never claim:

```text
Frontend works
```

unless it has been successfully started/built.

---

# 54. WHEN SOMETHING FAILS

If an error occurs:

1. Read the full error.
2. Identify root cause.
3. Fix the root cause.
4. Re-run the failing command.
5. Verify the fix.
6. Do not hide the error.
7. Do not replace the real solution with a fake placeholder.

If a dependency is incompatible:

* identify the incompatibility
* find a compatible version/alternative
* document the decision

---

# 55. DATASET AVAILABILITY

If the actual dataset files are not present:

Do NOT fabricate them.

Instead:

1. Create the expected directory.
2. Create a dataset download instruction.
3. Tell the user exactly where to place the file.
4. Make the pipeline ready to run after the dataset is provided.

The user will provide/download the datasets separately.

---

# 56. MODEL FALLBACK STRATEGY

If DistilBERT cannot be trained within the available local resources:

The application must still work using:

```text
TF-IDF + Logistic Regression
```

But the UI must clearly state:

```text
Transformer model unavailable
```

Do not pretend Logistic Regression is DistilBERT.

---

# 57. LOCAL RUN COMMANDS

The final README must provide exact commands.

Backend example:

```bash
cd backend
python app.py
```

Frontend example:

```bash
cd frontend
npm install
npm run dev
```

The final application should run locally using localhost URLs.

Example:

```text
Frontend:
http://localhost:5173

Backend:
http://localhost:5000
```

Use actual configured ports if different.

---

# 58. FINAL DEFINITION OF DONE

The project is complete only when:

[ ] Both datasets are supported

[ ] Dataset validation works

[ ] EDA works

[ ] Preprocessing works

[ ] TF-IDF works

[ ] Naive Bayes works

[ ] Logistic Regression works

[ ] SVM works

[ ] DistilBERT works OR its hardware limitation is documented

[ ] Actual evaluation metrics are generated

[ ] Confusion matrices are generated

[ ] Model comparison is generated

[ ] Error analysis exists

[ ] Flask API works

[ ] SQLite history works

[ ] React frontend works

[ ] Analytics dashboard works

[ ] Prediction works end-to-end

[ ] Invalid input is handled

[ ] Tests pass

[ ] README exists

[ ] Installation instructions work

[ ] No fake metrics exist

[ ] No fake predictions exist

[ ] No unnecessary secrets are committed

[ ] Git repository is clean

---

# 59. FINAL USER EXPERIENCE

A user should be able to open the web application and do this:

```text
1. Open SentimentSense AI

2. Enter:
   "I absolutely loved this movie!"

3. Click:
   Analyze

4. See:
   Positive

5. See:
   Confidence

6. See:
   Selected model

7. Compare:
   Logistic Regression
   SVM
   DistilBERT

8. View:
   Analytics

9. View:
   Prediction History
```

The application must feel like a real NLP product rather than a Jupyter Notebook wrapped in a webpage.

---

# 60. FIRST ACTION

When you start working on this project:

DO NOT immediately generate the entire project.

First:

1. Inspect the workspace.
2. Check Python version.
3. Check Node.js version.
4. Check Git.
5. Check available datasets.
6. Check whether a virtual environment exists.
7. Check whether React/Vite exists.
8. Check whether backend files already exist.
9. Create a concise implementation plan.
10. Create the project structure.
11. Create the README and configuration.
12. STOP and report the setup status.

After setup is verified, continue phase-by-phase.

---

# 61. COMMUNICATION FORMAT

After every major phase, report:

## Completed

List what was implemented.

## Files Created/Modified

List important files.

## Tests Run

List commands/tests.

## Results

Provide actual results.

## Issues

List unresolved issues honestly.

## Next Phase

State the next planned phase.

Do not overwhelm the user with unnecessary technical output.

---

# 62. MOST IMPORTANT PRINCIPLE

Build a project that is:

```text
CORRECT
+
REPRODUCIBLE
+
TESTED
+
ACADEMICALLY JUSTIFIABLE
+
VISUALLY PROFESSIONAL
+
ACTUALLY RUNNABLE
```

Do not optimize for the number of files.

Optimize for a reliable, understandable, high-quality NLP system.

---

## END OF MASTER INSTRUCTIONS
