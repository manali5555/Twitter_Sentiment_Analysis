![Twitter_Sentiment](/Twitter_Sentiment.JPG "")

# Twitter_Sentiment_Analysis

This project focuses on analyzing the sentiment of tweets as positive, negative, or neutral on a kaggle dataset related to various entities using a machine learning pipeline. It utilizes advanced NLP techniques and a gradient boosting classifier to predict the sentiment of tweets as positive, negative, or neutral.

## Project Overview

- Data preprocessing and exploration using Pandas and Seaborn.
- Sentence embeddings generated with Sentence Transformers.
- Sentiment classification with XGBoost.
- Model interpretation with LIME.

## Dataset
The dataset is sourced from kaggle which comprises tweets related to different entities, each annotated with sentiment labels. It is divided into training and validation sets, facilitating both the development and evaluation phases of the sentiment analysis model.

## Model Architecture
- **Sentence Transformers**: Used for converting tweets into meaningful numerical representations (embeddings) that capture the semantic essence of the text.
- **XGBoost Classifier**: A powerful, efficient, and scalable implementation of gradient boosting framework used for the classification task based on the generated embeddings.

## Preprocessing
The preprocessing steps include:
- Cleaning tweets to remove unnecessary characters and format the text.
- Renaming columns for easier access.
- Dropping missing values to maintain data integrity.
- Encoding sentiment labels into categorical codes for model training.

## Feature Extraction
- **Sentence Embeddings**: Tweets are transformed into high-dimensional vectors using Sentence Transformers, providing a robust feature set for classification.

## Training
The model is trained using the preprocessed and feature-engineered dataset, employing the XGBoost classifier with default parameters as a starting point.

## Evaluation
Model performance is evaluated on a validation set, with metrics such as accuracy, precision, recall, and F1-score being reported to assess the efficacy of the sentiment analysis.

## Usage
The project is structured into scripts that cover data preprocessing, EDA, feature extraction, model training, and evaluation. To execute the project:

1. Ensure all dependencies are installed.
2. Place the dataset in the `data/` directory.
3. Run the preprocessing script to clean and prepare the data.
4. Execute the feature extraction script to generate embeddings.
5. Train the model using the XGBoost classifier.
6. Evaluate the model's performance on the validation set.

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- sentence-transformers
- xgboost
- scikit-learn

## Installation

Ensure you have Python installed on your machine. Then, install the required packages using the following command:

```bash
pip install pandas numpy matplotlib seaborn sentence_transformers xgboost scikit-learn lime



**## Author**
Manali Ramchandani..

