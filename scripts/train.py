#!/usr/bin/env python3
"""
Training script for the subjectivity-value research project.
"""

import os
import sys
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import BertSubjectivityModel, TripletLoss, CombinedModel, create_tokenizer, encode_texts
from data_processing import load_training_data, load_test_data, load_value_categories
from utils import save_model_checkpoint, print_training_progress, calculate_triplet_accuracy


class SubjectivityDataset(Dataset):
    """Dataset class for subjectivity detection."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


def create_triplet_dataset(df, tokenizer, max_length=512):
    """Create triplet dataset for training."""
    # This is a simplified version - in practice, you'd implement
    # more sophisticated triplet mining based on your research
    
    texts = df['Premise'].tolist()
    labels = [eval(label) for label in df['simplified_Value_lvl2_ann']]
    
    dataset = SubjectivityDataset(texts, labels, tokenizer, max_length)
    return dataset


def train_triplet_model(model, dataloader, optimizer, criterion, device, epoch):
    """Train model with triplet loss."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings
        embeddings = model(input_ids, attention_mask)
        
        # For this example, we'll create simple triplets
        # In practice, you'd implement proper triplet mining
        batch_size = embeddings.size(0)
        if batch_size >= 3:
            anchor = embeddings[0::3]
            positive = embeddings[1::3]
            negative = embeddings[2::3]
            
            min_size = min(len(anchor), len(positive), len(negative))
            anchor = anchor[:min_size]
            positive = positive[:min_size]
            negative = negative[:min_size]
            
            if min_size > 0:
                loss = criterion(anchor, positive, negative)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                accuracy = calculate_triplet_accuracy(anchor, positive, negative)
                total_accuracy += accuracy
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    return avg_loss, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train subjectivity detection model')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='Directory containing data files')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pre-trained BERT model name')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--experiment', type=str, choices=['triplet', 'combined'], 
                       default='triplet', help='Experiment type')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_df = load_training_data(os.path.join(args.data_dir, 'train_.csv'))
    value_categories = load_value_categories(os.path.join(args.data_dir, 'value-categories.json'))
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Value categories: {len(value_categories)}")
    
    # Create tokenizer
    tokenizer = create_tokenizer(args.model_name)
    
    # Create dataset and dataloader
    dataset = create_triplet_dataset(train_df, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    if args.experiment == 'triplet':
        model = BertSubjectivityModel(model_name=args.model_name)
        criterion = TripletLoss(margin=1.0)
    elif args.experiment == 'combined':
        model = CombinedModel(
            bert_model_name=args.model_name,
            num_value_categories=len(value_categories)
        )
        criterion = TripletLoss(margin=1.0)
    
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    train_accuracies = []
    
    for epoch in range(args.epochs):
        avg_loss, avg_accuracy = train_triplet_model(
            model, dataloader, optimizer, criterion, device, epoch
        )
        
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Average Loss: {avg_loss:.4f}, "
              f"Average Accuracy: {avg_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth')
        save_model_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    main() 