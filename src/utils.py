"""
Utility functions for the subjectivity-value research project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
import torch


def calculate_fleiss_kappa(annotations: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa for inter-annotator agreement.
    
    Args:
        annotations: Array of shape (n_items, n_annotators)
        
    Returns:
        Fleiss' Kappa score
    """
    n_items, n_annotators = annotations.shape
    
    # Calculate observed agreement
    p_i = np.sum(annotations, axis=1) / n_annotators
    p_bar = np.mean(p_i)
    
    # Calculate expected agreement
    p_e = p_bar ** 2 + (1 - p_bar) ** 2
    
    # Calculate observed agreement
    p_o = np.mean(p_i ** 2 + (1 - p_i) ** 2)
    
    # Calculate Fleiss' Kappa
    if p_e == 1.0:
        return 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa


def evaluate_classification(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          labels: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate classification performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         labels: Optional[List[str]] = None,
                         title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    return fig


def plot_value_distribution(annotations_df: pd.DataFrame, 
                           value_categories: Dict[str, Dict],
                           title: str = "Value Distribution") -> plt.Figure:
    """
    Plot distribution of values in the dataset.
    
    Args:
        annotations_df: DataFrame with annotations
        value_categories: Dictionary of value categories
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Count occurrences of each value
    value_counts = {}
    for category, values in value_categories.items():
        for value in values.keys():
            value_counts[value] = value_counts.get(value, 0) + 1
    
    fig, ax = plt.subplots(figsize=(12, 8))
    values = list(value_counts.keys())
    counts = list(value_counts.values())
    
    bars = ax.bar(range(len(values)), counts)
    ax.set_xlabel('Values')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values, rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def calculate_subjectivity_scores(annotations_df: pd.DataFrame, 
                                 threshold: float = 0.5) -> np.ndarray:
    """
    Calculate subjectivity scores based on annotator disagreement.
    
    Args:
        annotations_df: DataFrame with multiple annotator columns
        threshold: Threshold for binary subjectivity classification
        
    Returns:
        Array of subjectivity scores
    """
    # Calculate variance across annotators for each example
    annotator_cols = [col for col in annotations_df.columns if 'annotator' in col.lower()]
    
    if len(annotator_cols) > 1:
        variances = annotations_df[annotator_cols].var(axis=1)
        subjectivity_scores = (variances > threshold).astype(int)
    else:
        # If only one annotator, use alternative method
        subjectivity_scores = np.zeros(len(annotations_df))
    
    return subjectivity_scores


def create_embeddings_visualization(embeddings: np.ndarray, 
                                   labels: np.ndarray,
                                   method: str = "tsne",
                                   title: str = "Embeddings Visualization") -> plt.Figure:
    """
    Create 2D visualization of embeddings using dimensionality reduction.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Labels for coloring points
        method: Dimensionality reduction method ('tsne', 'pca')
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6)
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    
    plt.colorbar(scatter, ax=ax)
    return fig


def save_model_checkpoint(model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int, 
                         loss: float,
                         filepath: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         filepath: str) -> Tuple[int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss


def calculate_triplet_accuracy(anchor: torch.Tensor, 
                              positive: torch.Tensor, 
                              negative: torch.Tensor) -> float:
    """
    Calculate accuracy for triplet loss (percentage of correct triplets).
    
    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings
        negative: Negative embeddings
        
    Returns:
        Triplet accuracy
    """
    dist_pos = torch.nn.functional.pairwise_distance(anchor, positive)
    dist_neg = torch.nn.functional.pairwise_distance(anchor, negative)
    
    correct = (dist_pos < dist_neg).float()
    accuracy = correct.mean().item()
    
    return accuracy


def print_training_progress(epoch: int, 
                           total_epochs: int,
                           batch_idx: int, 
                           total_batches: int,
                           loss: float, 
                           accuracy: Optional[float] = None) -> None:
    """
    Print training progress.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        batch_idx: Current batch index
        total_batches: Total number of batches
        loss: Current loss
        accuracy: Current accuracy (optional)
    """
    progress = (batch_idx + 1) / total_batches * 100
    
    if accuracy is not None:
        print(f"Epoch [{epoch+1}/{total_epochs}], "
              f"Batch [{batch_idx+1}/{total_batches}] ({progress:.1f}%), "
              f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    else:
        print(f"Epoch [{epoch+1}/{total_epochs}], "
              f"Batch [{batch_idx+1}/{total_batches}] ({progress:.1f}%), "
              f"Loss: {loss:.4f}")


def create_learning_curve_plot(train_losses: List[float], 
                              val_losses: List[float],
                              train_accuracies: Optional[List[float]] = None,
                              val_accuracies: Optional[List[float]] = None) -> plt.Figure:
    """
    Create learning curve plots.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_accuracies: Training accuracies (optional)
        val_accuracies: Validation accuracies (optional)
        
    Returns:
        Matplotlib figure
    """
    if train_accuracies is not None and val_accuracies is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracies
        ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    return fig 