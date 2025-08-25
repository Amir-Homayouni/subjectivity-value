# Subjectivity-Value Research Project

A machine learning research project investigating subjectivity in human value annotations using BERT embeddings and triplet loss methods.

## Overview

This project explores the relationship between inter-annotator disagreement and subjectivity in value classification tasks. The research hypothesis is that we can distinguish between representative (non-subjective) and minority (subjective) examples in value annotation datasets.

## Research Objectives

- **Experiment 2**: Distinguish between subjective and non-subjective value annotations using triplet loss with BERT embeddings
- **Experiment 3**: Combine triplet loss with Binary Cross-Entropy (BCE) loss for improved classification
- Investigate per-label subjectivity patterns across different human values
- Address class imbalance through data augmentation using paraphrasing

## Dataset

The project uses human value annotation data with:
- **Training set**: 6,484 annotated arguments (`train_.csv`)
- **Test set**: 1,818 annotated arguments (`test_.csv`) 
- **Value categories**: 21 value categories based on Schwartz's theory (`value-categories.json`)
- **Annotations**: Multi-annotator labels with subjectivity indicators

> **Note**: Large data files (`test_df.csv`, `annotations-level1.csv`) are excluded from this repository due to GitHub size limits. Please contact the author for access to the complete dataset.

### Value Categories
The dataset includes 21 human values organized into categories like:
- Self-direction (thought & action)
- Achievement, Power, Security
- Universalism, Benevolence, Tradition
- And more (see `value-categories.json`)

## Methodology

### Triplet Loss Approach
- **Anchor**: Text examples processed through BERT
- **Positive**: Examples with similar annotations (non-subjective)
- **Negative**: Examples with different annotations (subjective)

### Key Features
- BERT-based text embeddings
- Inter-annotator agreement analysis using Fleiss' Kappa
- Subjectivity detection at both global and per-label levels
- Data augmentation for imbalanced classes

## Installation

```bash
# Clone the repository
git clone https://github.com/Amir-Homayouni/subjectivity-value.git
cd subjectivity-value

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start with Jupyter Notebook
```bash
jupyter notebook Experiment2.ipynb
```

### Training Models
```bash
# Train triplet loss model
python scripts/train.py --experiment triplet --epochs 5 --batch_size 16

# Train combined model (triplet + BCE)
python scripts/train.py --experiment combined --epochs 10 --batch_size 8
```

### Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py --model_path checkpoints/final_model.pth --model_type triplet --visualize

# Generate predictions
python scripts/evaluate.py --model_path checkpoints/final_model.pth --model_type triplet --output_dir results/
```

### Programmatic Usage
```python
from src.models import BertSubjectivityModel, create_tokenizer
from src.data_processing import load_training_data
from src.utils import evaluate_classification

# Load data and model
tokenizer = create_tokenizer()
model = BertSubjectivityModel()
train_df = load_training_data('train_.csv')
```

## Project Structure

```
subjectivity-value/
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_processing.py    # Data loading and preprocessing utilities
│   ├── models.py            # BERT models and neural network architectures
│   └── utils.py             # Evaluation and visualization utilities
├── scripts/                 # Executable scripts
│   ├── train.py            # Training script for models
│   └── evaluate.py         # Evaluation and inference script
├── Experiment2.ipynb        # Original research notebook
├── train_.csv              # Training dataset (6,484 examples)
├── test_.csv               # Test dataset (1,818 examples)
├── value-categories.json   # Value category definitions (21 categories)
├── annotations-level1.csv  # Level 1 annotations
├── val_data.pkl           # Validation data
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation script
├── LICENSE               # MIT License
├── CITATION.cff         # Citation metadata
├── CONTRIBUTING.md      # Contribution guidelines
└── README.md           # This file
```

## Results

The research investigates whether models can effectively distinguish between subjective and objective value annotations. Key findings include challenges in distinguishing subjectivity when examples exhibit varying levels of subjectivity across different value dimensions.

## Dependencies

- PyTorch & Transformers (BERT models)
- scikit-learn (ML utilities)
- pandas & numpy (data processing)
- matplotlib & seaborn (visualization)

## License

This project is available for research and educational purposes.

## Citation

If you use this work, please cite:
```
@misc{homayouni2024subjectivity,
  author = {Amir Homayouni},
  title = {Subjectivity in Human Value Annotations},
  year = {2024},
  url = {https://github.com/Amir-Homayouni/subjectivity-value}
}
``` 