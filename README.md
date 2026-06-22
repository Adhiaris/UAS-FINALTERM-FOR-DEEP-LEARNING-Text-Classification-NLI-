# Task 1: Text Classification with DistilBERT

## Purpose

This repository contains the implementation of multi-class text classification using DistilBERT across three different datasets and tasks. The goal is to demonstrate how a pre-trained transformer encoder model can be fine-tuned for sequence classification problems of varying complexity.

## Project Overview

The notebook fine-tunes `distilbert-base-uncased` on three classification tasks:

- **Part A - AG News**: 4-class news topic classification (World, Sports, Business, Sci/Tech)
- **Part B - GoEmotions**: 28-class emotion classification on Reddit comments
- **Part C - MNLI**: 3-class Natural Language Inference (Entailment, Neutral, Contradiction)

Each part follows the same pipeline: dataset loading and EDA, tokenization, model fine-tuning with the HuggingFace Trainer API, evaluation, inference demo, and model saving.

## Models Used

| Model | Architecture | Parameters |
|-------|-------------|------------|
| distilbert-base-uncased | Encoder-only Transformer | 66M |

DistilBERT is a distilled version of BERT-base that is 40% smaller and 60% faster while retaining approximately 97% of BERT's performance. For classification, the `[CLS]` token representation is passed through a linear head to produce class logits.

## Results

### Part A - AG News (4 classes)

| Metric | Value |
|--------|-------|
| Eval Accuracy | 92.15% |
| Eval F1 (Macro) | 0.9224 |
| Eval Loss | 0.2479 |
| Train Samples | 20,000 |
| Eval Samples | 2,000 |

### Part B - GoEmotions (28 classes)

| Metric | Value |
|--------|-------|
| Eval Accuracy | 56.40% |
| Eval F1 (Macro) | 0.3855 |
| Eval Loss | 1.5306 |
| Train Samples | 15,000 |
| Eval Samples | 2,000 |

Note: Lower F1 on GoEmotions is expected due to class imbalance across 28 emotion categories and the inherent difficulty of fine-grained emotion classification.

### Part C - MNLI (3 classes)

| Metric | Expected Range |
|--------|---------------|
| Eval Accuracy (Matched) | 82-84% |
| Eval F1 (Macro) | 82-84% |
| Train Samples | 25,000 |

## Training Hyperparameters

| Parameter | Part A | Part B | Part C |
|-----------|--------|--------|--------|
| Epochs | 3 | 4 | 3 |
| Learning Rate | 2e-5 | 2e-5 | 2e-5 |
| Batch Size | 32 | 32 | 32 |
| Max Length | 128 | 128 | 128 |
| Warmup Steps | 200 | 200 | 200 |
| Weight Decay | 0.01 | 0.01 | 0.01 |

## How to Navigate

```
Task1_DistilBERT_All_Classification_NLI_FIXED.ipynb
|
|-- Step 1: Install Dependencies
|-- Step 2: Shared Setup (imports, device, shared metrics)
|
|-- PART A: AG News Classification
|   |-- A1. Load & Explore Dataset
|   |-- A2. Tokenization
|   |-- A3. Model & Training
|   |-- A4. Evaluation & Visualizations
|   |-- A5. Inference Demo
|   |-- A6. Save Model
|
|-- PART B: GoEmotions Emotion Classification
|   |-- B1. Load & Explore Dataset
|   |-- B2. Tokenization
|   |-- B3. Model & Training
|   |-- B4. Evaluation & Visualizations
|   |-- B5. Inference Demo
|   |-- B6. Save Model
|
|-- PART C: Natural Language Inference (MNLI)
    |-- C1. Load & Explore MNLI
    |-- C2. Tokenization (Sentence Pairs)
    |-- C3. Model & Training
    |-- C4. Evaluation (Matched vs Mismatched)
    |-- C5. Inference Demo
    |-- C6. Save Model
```

## Requirements

```
transformers
datasets
evaluate
accelerate
seaborn
torch
scikit-learn
```

## Notes

- AG News dataset uses labels 1-4 which are remapped to 0-3 before training to avoid CUDA assertion errors.
- MNLI uses `nyu-mll/multi_nli` as the dataset source. Column removal is done dynamically to avoid KeyError on missing `idx` column.
- For MNLI sentence-pair encoding, the format is `[CLS] premise [SEP] hypothesis [SEP]`.
