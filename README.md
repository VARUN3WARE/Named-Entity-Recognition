# Named Entity Recognition (NER) with Custom BILSTM Model and spaCy

This repository demonstrates two approaches for performing Named Entity Recognition (NER) on text data:
1. A custom BiLSTM (Bidirectional Long Short-Term Memory) model implemented in TensorFlow/Keras.
2. A quick implementation using spaCy's pre-trained model for comparison and efficiency.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
  - [1. Custom BiLSTM Model](#1-custom-bilstm-model)
  - [2. spaCy Pre-trained Model](#2-spacy-pre-trained-model)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to explore the implementation of Named Entity Recognition (NER) by:
- Building a custom BiLSTM model to classify words as named entities.
- Leveraging spaCy’s pre-trained model as a benchmark to evaluate the custom model's performance.

## Dataset

The dataset used for training the custom NER model is a labeled dataset containing tokens and their respective entity tags, such as `PER` (Person), `ORG` (Organization), `LOC` (Location), and `O` (Other). It is structured with columns:
- `Sentence #`: Indicates sentence grouping.
- `Word`: Each token/word.
- `POS`: Part-of-speech tag.
- `Tag`: Named entity label.

### Sample Data Structure

| Sentence # | Word        | POS  | Tag |
|------------|-------------|------|-----|
| Sentence 1 | Aman        | NNP  | PER |
| Sentence 1 | Kharwal     | NNP  | PER |
| Sentence 1 | works       | VB   | O   |
| Sentence 1 | at          | IN   | O   |
| Sentence 1 | Google      | NNP  | ORG |
| Sentence 2 | Steve Jobs  | NNP  | PER |

## Approach

### 1. Custom BiLSTM Model

- **Architecture**:
  - **Embedding Layer**: Creates word embeddings.
  - **Bidirectional LSTM**: Captures context from both directions.
  - **LSTM Layer**: Further context extraction for sequential data.
  - **TimeDistributed Dense Layer**: Outputs class probabilities for each token.

- **Training**:
  - The model is trained with a custom training loop over 25 epochs.
  - Data preprocessing includes token indexing, padding, and splitting into train, test, and validation sets.

### 2. spaCy Pre-trained Model

- **Model**: We use `spaCy`'s `en_core_web_sm` model, which includes pre-trained NER capabilities for English.
- **Usage**: Simply passing text through the model to get entities without additional training.

## Prerequisites

- Python 3.7+
- Libraries:
  - `spaCy`
  - `tensorflow`
  - `pandas`
  - `scikit-learn`
  - `numpy`

## Usage

1. **Custom BiLSTM Model**:
   - Run the provided notebook to execute each cell and train the model on the dataset.
   - After training, evaluate the model’s performance and visualize the results.

2. **spaCy Pre-trained Model**:
   - Load the `en_core_web_sm` model and pass a sample text for entity recognition.

### Example

To use spaCy's NER model:
```python
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
text = nlp("Steve Jobs founded Apple in California.")
displacy.render(text, style='ent', jupyter=True)
```

## Results

- The custom BiLSTM model's performance will depend on the quality and quantity of training data. Accuracy can be observed during the training loop.
- The spaCy model provides good results for general named entities but may lack specificity for domain-specific entities.

## Future Work

- Fine-tune the spaCy model with domain-specific data.
- Experiment with Transformer-based models (e.g., BERT) for NER.
- Further optimize the BiLSTM model with additional data or hyperparameter tuning.
- updating it for various languages support and more.

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and create a pull request with your changes.
 
