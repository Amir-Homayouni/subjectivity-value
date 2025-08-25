# Contributing to Subjectivity-Value Research Project

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing to the codebase.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/subjectivity-value.git
   cd subjectivity-value
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Development Setup

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep line length under 88 characters (Black formatter)

### Running Tests
```bash
# Run unit tests (when available)
pytest tests/

# Run linting
flake8 src/ scripts/
black --check src/ scripts/
```

## Project Structure

```
subjectivity-value/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── models.py          # BERT and neural network models
│   └── utils.py           # Utility functions
├── scripts/               # Training and evaluation scripts
│   ├── train.py          # Model training script
│   └── evaluate.py       # Model evaluation script
├── data/                 # Data files (gitignored)
├── checkpoints/          # Model checkpoints (gitignored)
├── results/              # Evaluation results (gitignored)
└── notebooks/            # Jupyter notebooks for exploration
```

## Making Changes

### For Bug Fixes
1. Create a new branch: `git checkout -b fix/issue-description`
2. Make your changes
3. Test your changes
4. Submit a pull request

### For New Features
1. Create a new branch: `git checkout -b feature/feature-name`
2. Implement your feature
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

### For Research Contributions
1. Document your methodology clearly
2. Include evaluation results
3. Add references to relevant literature
4. Update the README if necessary

## Code Organization

### Models (`src/models.py`)
- Implement new model architectures
- Follow the existing naming conventions
- Include proper docstrings with parameter descriptions

### Data Processing (`src/data_processing.py`)
- Add new data loading functions
- Ensure compatibility with existing data formats
- Handle edge cases and errors gracefully

### Utilities (`src/utils.py`)
- Add evaluation metrics
- Include visualization functions
- Maintain backward compatibility

## Research Guidelines

### Experiments
- Document all experimental settings
- Use reproducible random seeds
- Save model checkpoints for important runs
- Include comparison with baseline methods

### Data Handling
- Respect data privacy and licensing
- Do not commit large data files to git
- Use appropriate data splits for training/validation/testing

### Evaluation
- Use standard evaluation metrics
- Include statistical significance tests when appropriate
- Provide error bars or confidence intervals

## Submitting Changes

### Pull Request Process
1. Ensure your code follows the style guidelines
2. Update documentation if needed
3. Add a clear description of your changes
4. Reference any related issues

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense
- Keep the first line under 50 characters
- Include more detail in the body if needed

Example:
```
Add triplet mining strategy for hard negative selection

Implements online hard negative mining to improve triplet loss
training by selecting the most informative negative examples
during training.
```

## Getting Help

- Check existing issues on GitHub
- Create a new issue for bugs or feature requests
- Join discussions in the project's discussion forum
- Contact the maintainers for research collaboration

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

Thank you for contributing to advancing research in human value subjectivity detection! 