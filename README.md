# Fake News Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lilyrgb/fake-news-detection/blob/main/fake-news-detection.ipynb)

## Overview

This project builds an end-to-end machine learning pipeline for classifying news articles as real or fake. It combines TF-IDF representations of article titles and bodies with interpretable style features, compares linear classifiers, evaluates the selected model on a hold-out test set, supports individual predictions, and exports the complete pipeline for deployment.

## Project Objective

The goal is to demonstrate a transparent and reproducible text-classification workflow while highlighting the difference between strong performance on a benchmark dataset and reliable generalization to unseen publishers, topics, and time periods.

## Repository Contents

| File | Description |
| --- | --- |
| `fake-news-detection.ipynb` | Clean English Google Colab notebook containing the complete analysis and modeling workflow. |
| `fake-news-articles.zip` | Compressed `Fake.csv` dataset containing 23,481 articles labeled as fake. |
| `true-news-articles.zip` | Compressed `True.csv` dataset containing 21,417 articles labeled as real. |

The original dataset archive exceeded GitHub's browser upload limit, so its two CSV files are provided as separate ZIP archives without changing their contents.

## Workflow

### 1. Data Loading and Exploration

- Download and extract the source dataset.
- Combine the real and fake article tables.
- Inspect missing values, empty text, duplicates, subjects, and class balance.
- Assign explicit binary labels for modeling.

### 2. Data Cleaning

- Standardize text fields.
- Remove duplicate title-and-body combinations.
- Exclude empty article bodies.
- Preserve a reproducible cleaned dataset for downstream modeling.

The raw dataset contains 44,898 articles. After cleaning and duplicate removal, the notebook retains 38,658 observations.

### 3. Feature Engineering

- Normalize article titles and bodies.
- Create TF-IDF word and bigram features.
- Calculate title and article lengths.
- Measure word counts, punctuation, uppercase ratio, and suspicious-keyword frequency.

### 4. Leakage-Aware Design

The dataset's `subject` categories are closely associated with the source collections and target labels. The published notebook intentionally excludes this field from predictive features to reduce direct source leakage.

### 5. Model Selection

The notebook compares:

- Logistic Regression
- Linear Support Vector Machine

Models are ranked using validation F1 score. The selected pipeline is then evaluated once on a stratified hold-out test set.

### 6. Evaluation and Deployment

- Accuracy, precision, recall, F1, ROC-AUC where available, and confusion matrix
- Misclassification review
- Reusable inference helper
- Export of the complete preprocessing and classification pipeline as `fake_news_pipeline.pkl`

## Important Interpretation

Very high performance on this dataset must be interpreted cautiously. Random row-level splits may preserve publisher, writing-style, and time-period signals across training and test data. A production-quality assessment should include source-separated and time-separated validation.

The model should be treated as a screening aid, not as independent proof that an article is true or false.

## Technologies

- Python 3
- Google Colab
- pandas and NumPy
- scikit-learn
- NLTK
- TF-IDF
- Logistic Regression
- Linear SVM
- Matplotlib and Seaborn
- joblib

## Dataset

The source dataset contains:

- `Fake.csv`: 23,481 fake-news articles
- `True.csv`: 21,417 real-news articles

Each table includes article title, body text, subject, and publication date. To use the repository copies locally, extract both ZIP archives into the same directory. The Colab notebook can also download the complete original archive automatically.

## How to Run

1. Click the **Open in Colab** badge above.
2. Run the notebook cells in order.
3. Allow the dataset download and NLTK resource setup to complete.
4. Review the data-quality and leakage checks.
5. Compare validation results and inspect the final hold-out evaluation.
6. Use the inference helper or export the pipeline for an application.

## Limitations

- The dataset represents a limited set of publishers and historical topics.
- Random splits do not fully measure performance on unseen sources.
- Text-only classification cannot detect manipulated images, audio, or video.
- Model predictions require human review and independent source verification.

## Publication Notes

- Saved execution outputs and personal notebook metadata were removed.
- Python syntax was validated across all code cells.
- The source-specific `subject` field was excluded from predictive features.
- All repository documentation and notebook content are in English.
