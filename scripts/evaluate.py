#!/usr/bin/env python3
"""
Evaluation script for the subjectivity-value research project.
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import BertSubjectivityModel, CombinedModel, create_tokenizer
from data_processing import load_test_data, load_value_categories
from utils import evaluate_classification, plot_confusion_matrix, create_embeddings_visualization


def load_model(model_path, model_type, model_name='bert-base-uncased', num_value_categories=21):
    """Load trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'triplet':
        model = BertSubjectivityModel(model_name=model_name)
    elif model_type == 'combined':
        model = CombinedModel(
            bert_model_name=model_name,
            num_value_categories=num_value_categories
        )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device


def extract_embeddings(model, texts, tokenizer, device, batch_size=32):
    """Extract embeddings from texts using the trained model."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encoded = tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            if hasattr(model, 'bert_model'):
                # Combined model
                outputs = model(input_ids, attention_mask, mode='embedding')
                batch_embeddings = outputs['embeddings']
            else:
                # Triplet model
                batch_embeddings = model(input_ids, attention_mask)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def evaluate_subjectivity_detection(embeddings, labels, threshold=0.5):
    """Evaluate subjectivity detection performance."""
    # Simple evaluation using embedding similarity
    # In practice, you'd implement a more sophisticated evaluation
    
    # Calculate pairwise distances
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(embeddings)
    
    # Predict subjectivity based on average distance to other examples
    avg_distances = np.mean(distances, axis=1)
    predictions = (avg_distances > threshold).astype(int)
    
    # Evaluate if we have ground truth labels
    if labels is not None:
        metrics = evaluate_classification(labels, predictions)
        return predictions, metrics
    
    return predictions, None


def main():
    parser = argparse.ArgumentParser(description='Evaluate subjectivity detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['triplet', 'combined'],
                       required=True, help='Type of model')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory containing data files')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pre-trained BERT model name')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading test data...")
    test_df = load_test_data(os.path.join(args.data_dir, 'test_.csv'))
    value_categories = load_value_categories(os.path.join(args.data_dir, 'value-categories.json'))
    
    print(f"Test data shape: {test_df.shape}")
    
    # Load model
    print("Loading model...")
    model, device = load_model(
        args.model_path, 
        args.model_type, 
        args.model_name,
        len(value_categories)
    )
    
    # Create tokenizer
    tokenizer = create_tokenizer(args.model_name)
    
    # Extract embeddings
    print("Extracting embeddings...")
    texts = test_df['Premise'].tolist()
    embeddings = extract_embeddings(model, texts, tokenizer, device)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Evaluate subjectivity detection
    print("Evaluating subjectivity detection...")
    
    # For this example, we'll create dummy subjectivity labels
    # In practice, you'd have ground truth subjectivity labels
    dummy_labels = np.random.randint(0, 2, len(texts))
    
    predictions, metrics = evaluate_subjectivity_detection(embeddings, dummy_labels)
    
    # Print results
    if metrics:
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        # Save detailed classification report
        report = classification_report(dummy_labels, predictions, output_dict=True)
        report_path = os.path.join(args.output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Classification report saved: {report_path}")
    
    # Save embeddings
    embeddings_path = os.path.join(args.output_dir, 'embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved: {embeddings_path}")
    
    # Save predictions
    predictions_df = test_df.copy()
    predictions_df['subjectivity_prediction'] = predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved: {predictions_path}")
    
    # Create visualizations if requested
    if args.visualize:
        print("Creating visualizations...")
        
        # Confusion matrix
        if metrics:
            fig = plot_confusion_matrix(
                dummy_labels, 
                predictions, 
                labels=['Non-subjective', 'Subjective'],
                title='Subjectivity Detection Confusion Matrix'
            )
            confusion_path = os.path.join(args.output_dir, 'confusion_matrix.png')
            fig.savefig(confusion_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {confusion_path}")
        
        # Embeddings visualization
        fig = create_embeddings_visualization(
            embeddings, 
            predictions,
            method='tsne',
            title='Subjectivity Embeddings (t-SNE)'
        )
        tsne_path = os.path.join(args.output_dir, 'embeddings_tsne.png')
        fig.savefig(tsne_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved: {tsne_path}")
        
        # PCA visualization
        fig = create_embeddings_visualization(
            embeddings, 
            predictions,
            method='pca',
            title='Subjectivity Embeddings (PCA)'
        )
        pca_path = os.path.join(args.output_dir, 'embeddings_pca.png')
        fig.savefig(pca_path, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved: {pca_path}")
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 