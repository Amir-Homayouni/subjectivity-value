"""
Model implementations for the subjectivity-value research project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple, Optional


class BertSubjectivityModel(nn.Module):
    """
    BERT-based model for subjectivity detection using triplet loss.
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 embedding_dim: int = 768,
                 dropout_rate: float = 0.1):
        """
        Initialize the BERT subjectivity model.
        
        Args:
            model_name: Pre-trained BERT model name
            embedding_dim: Dimension of BERT embeddings
            dropout_rate: Dropout rate for regularization
        """
        super(BertSubjectivityModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding_dim = embedding_dim
        
        # Optional projection layer
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Pooled embeddings
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token embedding
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Optional projection
        embeddings = self.projection(pooled_output)
        
        return embeddings


class TripletLoss(nn.Module):
    """
    Triplet loss for subjectivity detection.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (similar)
            negative: Negative embeddings (dissimilar)
            
        Returns:
            Triplet loss value
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class SubjectivityClassifier(nn.Module):
    """
    Binary classifier for subjectivity detection on top of BERT embeddings.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 dropout_rate: float = 0.1):
        """
        Initialize the classifier.
        
        Args:
            embedding_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(SubjectivityClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classifier.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Classification logits
        """
        return self.classifier(embeddings)


class CombinedModel(nn.Module):
    """
    Combined model using triplet loss and BCE loss.
    """
    
    def __init__(self, 
                 bert_model_name: str = "bert-base-uncased",
                 embedding_dim: int = 768,
                 hidden_dim: int = 256,
                 num_value_categories: int = 21,
                 dropout_rate: float = 0.1):
        """
        Initialize the combined model.
        
        Args:
            bert_model_name: Pre-trained BERT model name
            embedding_dim: BERT embedding dimension
            hidden_dim: Hidden layer dimension
            num_value_categories: Number of value categories
            dropout_rate: Dropout rate
        """
        super(CombinedModel, self).__init__()
        
        self.bert_model = BertSubjectivityModel(
            model_name=bert_model_name,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
        # Classifier for each value category
        self.value_classifiers = nn.ModuleList([
            SubjectivityClassifier(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_classes=2,
                dropout_rate=dropout_rate
            ) for _ in range(num_value_categories)
        ])
        
        self.triplet_loss = TripletLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                value_labels: Optional[torch.Tensor] = None,
                mode: str = "embedding") -> Dict[str, torch.Tensor]:
        """
        Forward pass through the combined model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            value_labels: Value labels for training
            mode: Forward mode ('embedding', 'classification', 'both')
            
        Returns:
            Dictionary with embeddings and/or classifications
        """
        # Get BERT embeddings
        embeddings = self.bert_model(input_ids, attention_mask)
        
        outputs = {"embeddings": embeddings}
        
        if mode in ["classification", "both"]:
            # Get classifications for each value category
            classifications = []
            for classifier in self.value_classifiers:
                logits = classifier(embeddings)
                classifications.append(logits)
            
            outputs["classifications"] = torch.stack(classifications, dim=1)
        
        return outputs


def create_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizer:
    """
    Create BERT tokenizer.
    
    Args:
        model_name: Pre-trained BERT model name
        
    Returns:
        BERT tokenizer
    """
    return BertTokenizer.from_pretrained(model_name)


def encode_texts(texts: List[str], 
                tokenizer: BertTokenizer,
                max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Encode texts using BERT tokenizer.
    
    Args:
        texts: List of text strings
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with encoded inputs
    """
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    return encoded 