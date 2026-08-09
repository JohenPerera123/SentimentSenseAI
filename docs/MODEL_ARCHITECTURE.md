# Model Architecture

## 1. Traditional ML Models
The initial phases use TF-IDF vectorization coupled with three classical algorithms:
- **Naive Bayes**: A fast probabilistic baseline for text classification.
- **Logistic Regression**: A robust linear model providing calibrated probabilities.
- **Linear SVM**: Maximizes the margin between classes for high-dimensional text data.

## 2. Transformer Model: DistilBERT
In Phase 4, we implemented `distilbert-base-uncased`:
- **Architecture**: A smaller, faster, cheaper, lighter version of BERT. It has 40% fewer parameters than `bert-base-uncased`, runs 60% faster, while preserving over 95% of BERT's performance as measured on the GLUE benchmark.
- **Tokenizer**: Subword tokenization using WordPiece. Max sequence length is configured to 128 tokens for efficiency.
- **Head**: Sequence Classification head with 2 outputs (Negative vs Positive).
- **Service Integration**: The `ModelService` router abstracts away the complexities of inference between the traditional TF-IDF pipeline and the PyTorch Transformer pipeline.

## Model Selection Strategy
The overarching architecture leverages a dynamic service layer (`src/models/model_service.py`) that handles routing predictions to the correct model. This guarantees that API callers can dynamically select "DistilBERT" or "Logistic Regression" without worrying about underlying text preprocessing differences.
