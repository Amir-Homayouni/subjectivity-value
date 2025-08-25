"""
Subjectivity-Value Research Project

A machine learning research project investigating subjectivity in human value 
annotations using BERT embeddings and triplet loss methods.
"""

__version__ = "1.0.0"
__author__ = "Amir Homayouni"
__email__ = "your.email@example.com"

from . import data_processing
from . import models
from . import utils

__all__ = ["data_processing", "models", "utils"] 