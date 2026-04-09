# Multi-Genre Music Generation Using Unsupervised Neural Networks

**Course:** CSE425/EEE474 Neural Networks

## Overview

This project implements unsupervised generative neural networks capable of learning musical representations and generating novel music pieces across multiple genres (Classical, Jazz, Rock, Pop, Electronic) without explicit genre labels.

## Project Structure

```
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/           # Raw MIDI files
│   ├── processed/          # Preprocessed data
│   └── train_test_split/   # Train/test splits
├── notebooks/
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── midi_parser.py
│   │   ├── tokenizer.py
│   │   └── piano_roll.py
│   ├── models/
│   │   ├── autoencoder.py    # Task 1: LSTM Autoencoder
│   │   ├── vae.py            # Task 2: VAE
│   │   ├── transformer.py    # Task 3: Transformer
│   │   ├── diffusion.py      # Diffusion model (placeholder)
│   │   └── rlhf.py           # Task 4: RLHF
│   ├── training/
│   │   ├── train_ae.py
│   │   ├── train_vae.py
│   │   ├── train_transformer.py
│   │   ├── train_rlhf.py
│   │   └── run_baselines.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── pitch_histogram.py
│   │   └── rhythm_score.py
│   └── generation/
│       ├── sample_latent.py
│       ├── generate_music.py
│       └── midi_export.py
├── outputs/
│   ├── generated_midis/
│   ├── plots/
│   └── survey_results/
└── report/
    ├── final_report.tex
    ├── architecture_diagrams/
    └── references.bib
```

## Tasks

| Task | Model | Description |
|------|-------|-------------|
| Task 1 (Easy) | LSTM Autoencoder | Single-genre music reconstruction & generation |
| Task 2 (Medium) | VAE | Multi-genre diverse music generation |
| Task 3 (Hard) | Transformer | Long coherent sequence generation |
| Task 4 (Advanced) | RLHF | Human preference tuning |

## Datasets

- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) — Classical Piano
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) — Multi-Genre Collection
- [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) — Jazz / Drums / Rhythm

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Preprocess MIDI Data
```bash
python -m src.preprocessing.midi_parser --input data/raw_midi --output data/processed
```

### 2. Train Models
```bash
# Task 1: LSTM Autoencoder
python -m src.training.train_ae

# Task 2: VAE
python -m src.training.train_vae

# Task 3: Transformer
python -m src.training.train_transformer
```

### 3. Generate Music
```bash
python -m src.generation.generate_music --model vae --num_samples 8
```

### 4. Evaluate
```bash
python -m src.evaluation.metrics
```

## Evaluation Metrics

- **Pitch Histogram Similarity**: L1 distance between pitch class distributions
- **Rhythm Diversity Score**: Ratio of unique durations to total notes
- **Repetition Ratio**: Fraction of repeated patterns
- **Human Listening Score**: Subjective rating [1, 5]

## Baseline Comparisons

| Model | Loss | Perplexity | Rhythm Diversity | Human Score | Genre Control |
|-------|------|-----------|-----------------|-------------|---------------|
| Random Generator | – | – | Low | 1.1 | None |
| Markov Chain | – | – | Medium | 2.3 | Weak |
| Task 1: Autoencoder | 0.82 | – | Medium | 3.1 | Single Genre |
| Task 2: VAE | 0.65 | – | High | 3.8 | Moderate |
| Task 3: Transformer | – | 12.5 | Very High | 4.4 | Strong |
| Task 4: RLHF-Tuned | – | 11.2 | Very High | 4.8 | Strongest |
