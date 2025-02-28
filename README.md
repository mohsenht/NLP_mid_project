# Sentiment Analysis Using BiLSTM with and without Pretrained Word Embeddings

This project applies **Bidirectional LSTM (BiLSTM)** models for sentiment analysis, comparing **self-trained word embeddings** with **pretrained GloVe embeddings**.

## Dataset Download
- **Sentiment140 Dataset**: Download from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Unzip the dataset and place it in the project directory.

## Pretrained Word Embeddings
- **GloVe Embeddings**: Download from [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/)
- Use **glove.6B.100d.txt** and unzip it in the project directory.

## Setup
Ensure both the **dataset and embeddings** are placed in the correct directory before running the scripts.

## Running the Model
Run the following commands separately to train both models:

```bash
python model_with_self_embedding.py
python model_with_GloVe.py
