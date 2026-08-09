# Final Held-Out Test Evaluation

## Purpose
The test set was kept completely unseen during model development, hyperparameter selection, and TF-IDF fitting. This ensures the following metrics represent true, unbiased generalization performance on novel data.

## Dataset Splits
### IMDb
- Train count: 40000
- Validation count: 5000 
- Test count: 5000

### Twitter
- Train count: 40000
- Validation count: 5000
- Test count: 5000

## Leakage Prevention
- TF-IDF fitted on training data only.
- Validation and test use `transform()` only.
- Models were not trained on test data.

## IMDb Final Test Results
| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Baseline | 0.5000 | 0.2500 | 0.5000 | 0.3333 |
| Naive Bayes | 0.8814 | 0.8814 | 0.8814 | 0.8814 |
| Logistic Regression | 0.9074 | 0.9074 | 0.9074 | 0.9074 |
| Linear SVM | 0.9074 | 0.9074 | 0.9074 | 0.9074 |

## Twitter Final Test Results
| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Baseline | 0.5000 | 0.2500 | 0.5000 | 0.3333 |
| Naive Bayes | 0.7636 | 0.7644 | 0.7636 | 0.7634 |
| Logistic Regression | 0.7774 | 0.7774 | 0.7774 | 0.7774 |
| Linear SVM | 0.7652 | 0.7652 | 0.7652 | 0.7652 |

## Best Models
- Best IMDb traditional model: Logistic Regression
- Best Twitter traditional model: Logistic Regression

## Validation vs Test
Validation performance generally matches test performance within a reasonable margin. There is no strong evidence of aggressive overfitting, as the models hold their generalization capability well across both domains.

## Error Analysis
Manual review of error files (`reports/imdb_test_errors.csv` and `reports/twitter_test_errors.csv`) highlights persistent issues with sarcasm, deeply nested negations, and very short ambiguous phrases where traditional TF-IDF bag-of-words completely fails to capture the sequence and context.

## Conclusion
Traditional models form a highly capable baseline. Logistic Regression strictly edges out linear SVM and Naive Bayes in these sentiment contexts. However, structural and contextual errors remain unresolved by bag-of-words TF-IDF representations.
