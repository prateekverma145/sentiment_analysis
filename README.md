# Sentiment Analysis Using Bidirectional GRU with Attention

This project implements sentiment analysis using a Bidirectional Gated Recurrent Unit (Bi-GRU) model combined with an Attention Mechanism. The goal is to leverage the advantages of both architectures to capture long-range dependencies in the text and give focus to the most important words when predicting sentiment.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Attention Mechanism](#attention-mechanism)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Analysis](#results-and-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Overview
Sentiment analysis is the task of determining whether a text conveys a positive or negative sentiment. In this project, we use a **Bidirectional GRU** combined with an **Attention Mechanism** to enhance the model's ability to understand the sequence of words in both forward and backward directions and emphasize the parts of the input sequence that are most relevant to sentiment classification.

The main objectives of this project are:
- To implement and train a Bidirectional GRU model with Attention for sentiment classification.
- To evaluate its performance on a labeled sentiment dataset using common evaluation metrics.
- To analyze the contribution of the Attention Mechanism in improving sentiment classification performance.

## Model Architecture

### 1. **Bidirectional GRU**
GRU is a type of recurrent neural network (RNN) that addresses the vanishing gradient problem and is capable of learning long-term dependencies. The **Bidirectional GRU** processes the text in both forward and backward directions, which allows it to better understand context and the relationships between words in a sentence.

### 2. **Attention Mechanism**
The Attention Mechanism allows the model to focus on the most relevant parts of the input sequence, dynamically weighing the importance of each word. This helps the model prioritize significant words when making predictions, which is especially useful in tasks like sentiment analysis where certain words can heavily influence sentiment.

#### Bi-GRU with Attention Architecture:
- **Embedding Layer**: Converts the words into dense vectors.
- **Bidirectional GRU Layer**: Processes the sequence in both forward and backward directions.
- **Attention Layer**: Adds attention scores to highlight important words in the sequence.
- **Fully Connected Layer**: Outputs the sentiment prediction (positive or negative).

## Attention Mechanism
In a traditional GRU, all hidden states contribute equally to the output. However, the **Attention Mechanism** assigns different weights to each hidden state, allowing the model to attend to more important words. This improves the model's interpretability and often enhances performance by giving more importance to crucial parts of the input.

The Attention Mechanism computes:
1. **Attention Weights**: For each word in the sequence.
2. **Context Vector**: A weighted sum of the hidden states, focusing on the important words.
3. **Final Output**: The sentiment classification based on the context vector.

## Evaluation Metrics
The following metrics are used to evaluate the model's performance:
- **Accuracy**: Percentage of correct predictions over the total predictions.
- **Precision**: Ratio of true positive predictions to all positive predictions.
- **Recall**: Ratio of true positive predictions to all actual positive examples.
- **F1 Score**: Harmonic mean of precision and recall, providing a balanced measure of performance.

```python
Evaluation Metrics:
    - Accuracy: acc_score
    - Precision: precision_score
    - Recall: recall_score
    - F1 Score: fscore_score
