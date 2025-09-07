# Filler's Transformer – Intent Classification with DistilBERT

This project implements an **intent classification model** using the **SNIPS dataset** and **DistilBERT transformer architecture**. It is a deep learning project focusing on natural language understanding (NLU) for intent detection.

---

## Project Overview

The model classifies user intents based on input text using a fine-tuned **DistilBERT** transformer with TensorFlow.

### Features
- Dataset preparation with **Hugging Face Datasets**
- Transformer-based architecture with a custom classification head
- Training, validation, and testing pipelines
- Save/load model functionality for production reuse

---

## Dataset

- **Dataset used:** [SNIPS Joint Intent Dataset](https://huggingface.co/datasets/bkonkle/snips-joint-intent)
- Contains natural language utterances mapped to their corresponding intents.

---

## Project Structure
```
├── filler_s_transformer.py # Main project script
├── filler_s_transformer.h5 # Model weights (generated after training)
├── README.md # Project documentation
```

## Installation & Requirements

### Dependencies
- Python 3.9+
- TensorFlow 2.x
- Hugging Face Transformers
- Datasets
- Pandas

## Install all dependencies:
```bash
pip install tensorflow transformers datasets pandas
```

## Model Architecture

1) Base Model: DistilBERT (distilbert-base-uncased)
2) Classification Head: Dense layer with softmax activation
3) Loss Function: Sparse Categorical Crossentropy
4) Optimizer: Adam (learning rate = 5e-5)