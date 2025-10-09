"""
M6ANet with Multiple Instance Learning (MIL) Framework

Multiple Instance Learning Framework:
- Each site is a "bag" containing multiple "instances" (reads)
- Bag label: 1 if modified (m6A present), 0 if unmodified
- Instance labels: Unknown (we don't know which specific reads show modification)
- MIL assumption: A bag is positive if at least one instance is positive

Three MIL Approaches Implemented:
1. Attention-based MIL (Ilse et al., 2018)
2. Instance-level MIL with max/mean pooling
3. Embedded-space MIL with gated attention

Key Advantages:
- Explicitly models the bag-instance relationship
- Learns which reads are most informative
- Better handles noisy/variable read quality
- Provides interpretability through attention weights
"""

import pandas as pd
import numpy as np
import gzip
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for MIL-based m6A detection"""
    
    # Data parameters
    N_READS_PER_SITE = 20  # Max reads per bag
    INPUT_DIM = 9
    
    # MIL Model Selection
    MIL_TYPE = 'gated_attention'  # Options: 'attention', 'instance', 'gated_attention'
    
    # Model architecture
    HIDDEN_DIM = 128
    ATTENTION_DIM = 64
    N_CLASSES = 1  # Binary classification
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 64  # Batch of bags
    EPOCHS = 20

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # MIL-specific parameters
    INSTANCE_LOSS_WEIGHT = 0.3  # Weight for instance-level loss
    BAG_LOSS_WEIGHT = 0.7       # Weight for bag-level loss
    
    # Focal Loss for imbalanced data
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0
    
    # Training optimization
    MIXED_PRECISION = True
    GRADIENT_CLIP = 1.0
    NUM_WORKERS = 4
    
    # Early stopping
    PATIENCE = 15
    N_FOLDS = 5
    
    # Paths
    DATA_FILE = 'dataset_0/dataset0.json.gz'
    LABELS_FILE = 'dataset_0/data.info.labelled'
    OUTPUT_DIR = 'results_mil_m6anet_gated'
    CHECKPOINT_DIR = 'checkpoints_mil'
    
    LOG_INTERVAL = 10


# ============================================================================
# MULTIPLE INSTANCE LEARNING MODELS
# ============================================================================

class AttentionMIL(nn.Module):
    """
    Attention-based MIL (Ilse et al., 2018)
    
    Uses attention mechanism to weight instances in a bag.
    Learns which reads are most informative for modification detection.
    
    Architecture:
    1. Instance-level embedding
    2. Attention-based pooling
    3. Bag-level classification
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Instance-level feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(config.ATTENTION_DIM, 1)
        )
        
        # Bag-level classifier (output logits, not probabilities)
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.N_CLASSES)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_instances, input_dim)
        Returns:
            bag_prediction: (batch_size, n_classes) - LOGITS
            attention_weights: (batch_size, n_instances)
        """
        batch_size, n_instances, _ = x.shape
        
        # Reshape for batch norm
        x = x.view(-1, self.config.INPUT_DIM)
        
        # Extract instance features
        instance_features = self.feature_extractor(x)  # (batch*n_instances, hidden_dim)
        instance_features = instance_features.view(batch_size, n_instances, -1)
        
        # Compute attention weights
        attention_scores = self.attention(instance_features.view(-1, self.config.HIDDEN_DIM))
        attention_scores = attention_scores.view(batch_size, n_instances, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, n_instances, 1)
        
        # Attention-weighted pooling
        bag_representation = torch.sum(instance_features * attention_weights, dim=1)  # (batch_size, hidden_dim)
        
        # Bag-level prediction (logits)
        bag_logits = self.classifier(bag_representation).squeeze(-1)
        
        return bag_logits, attention_weights.squeeze(-1)
    
    def calculate_objective(self, x, bag_labels):
        """
        Calculate MIL objective
        
        Args:
            x: (batch_size, n_instances, input_dim)
            bag_labels: (batch_size,)
        """
        bag_logits, attention_weights = self(x)
        
        # Bag-level loss (using logits)
        if self.config.USE_FOCAL_LOSS:
            bag_loss = focal_loss_with_logits(bag_logits, bag_labels, 
                                             alpha=self.config.FOCAL_ALPHA, 
                                             gamma=self.config.FOCAL_GAMMA)
        else:
            bag_loss = F.binary_cross_entropy_with_logits(bag_logits, bag_labels)
        
        # Return logits for predictions (will apply sigmoid later)
        return bag_loss, bag_logits, attention_weights


class InstanceMIL(nn.Module):
    """
    Instance-level MIL with max/mean pooling
    
    Predicts instance-level probabilities, then aggregates:
    - Positive bag: max(instance_probs) (at least one positive instance)
    - Negative bag: mean(instance_probs) (all instances negative)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Instance-level classifier (output logits)
        self.instance_classifier = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_instances, input_dim)
        Returns:
            bag_prediction: (batch_size,) - LOGITS
            instance_predictions: (batch_size, n_instances) - LOGITS
        """
        batch_size, n_instances, _ = x.shape
        
        # Reshape for batch processing
        x = x.view(-1, self.config.INPUT_DIM)
        
        # Instance-level predictions (logits)
        instance_logits = self.instance_classifier(x).squeeze(-1)
        instance_logits = instance_logits.view(batch_size, n_instances)
        
        # Bag-level prediction: max pooling on probabilities
        instance_probs = torch.sigmoid(instance_logits)
        bag_prob, _ = torch.max(instance_probs, dim=1)
        
        # Convert back to logits for loss calculation
        bag_logits = torch.log(bag_prob / (1 - bag_prob + 1e-7))
        
        return bag_logits, instance_logits
    
    def calculate_objective(self, x, bag_labels):
        """
        Calculate MIL objective with instance-level supervision
        """
        bag_logits, instance_logits = self(x)
        
        # Bag-level loss
        if self.config.USE_FOCAL_LOSS:
            bag_loss = focal_loss_with_logits(bag_logits, bag_labels, 
                                             alpha=self.config.FOCAL_ALPHA, 
                                             gamma=self.config.FOCAL_GAMMA)
        else:
            bag_loss = F.binary_cross_entropy_with_logits(bag_logits, bag_labels)
        
        # Instance-level loss (pseudo-labeling)
        instance_loss = 0
        for i, label in enumerate(bag_labels):
            if label == 1:
                # Positive bag: max instance should be high
                max_instance_logit = torch.max(instance_logits[i])
                instance_loss += F.binary_cross_entropy_with_logits(
                    max_instance_logit, torch.ones_like(max_instance_logit)
                )
            else:
                # Negative bag: all instances should be low
                instance_loss += F.binary_cross_entropy_with_logits(
                    instance_logits[i], torch.zeros_like(instance_logits[i])
                )
        
        instance_loss /= len(bag_labels)
        
        # Combined loss
        total_loss = (self.config.BAG_LOSS_WEIGHT * bag_loss + 
                     self.config.INSTANCE_LOSS_WEIGHT * instance_loss)
        
        return total_loss, bag_logits, instance_logits


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL (Ilse et al., 2018)
    
    Uses gated attention mechanism with two pathways:
    - Attention pathway: learns what to attend to
    - Gate pathway: learns how much to attend
    
    More expressive than standard attention.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Instance-level feature extractor
        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
        )
        
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
        )
        
        # Gated attention
        self.attention_V = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.ATTENTION_DIM),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.ATTENTION_DIM),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(config.ATTENTION_DIM, 1)
        
        # Bag-level classifier (output logits)
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.N_CLASSES)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_instances, input_dim)
        Returns:
            bag_prediction: (batch_size,) - LOGITS
            attention_weights: (batch_size, n_instances)
        """
        batch_size, n_instances, _ = x.shape
        
        # Extract features
        x = x.view(-1, self.config.INPUT_DIM)
        h = self.feature_extractor_part1(x)
        h = self.feature_extractor_part2(h)
        h = h.view(batch_size, n_instances, -1)
        
        # Gated attention
        h_flat = h.view(-1, self.config.HIDDEN_DIM)
        A_V = self.attention_V(h_flat)  # Attention pathway
        A_U = self.attention_U(h_flat)  # Gate pathway
        A = self.attention_weights(A_V * A_U)  # Element-wise gating
        
        A = A.view(batch_size, n_instances)
        attention_weights = F.softmax(A, dim=1)
        
        # Attention-weighted pooling
        bag_representation = torch.sum(h * attention_weights.unsqueeze(-1), dim=1)
        
        # Bag-level prediction (logits)
        bag_logits = self.classifier(bag_representation).squeeze(-1)
        
        return bag_logits, attention_weights
    
    def calculate_objective(self, x, bag_labels):
        """Calculate MIL objective"""
        bag_logits, attention_weights = self(x)
        
        if self.config.USE_FOCAL_LOSS:
            loss = focal_loss_with_logits(bag_logits, bag_labels, 
                                        alpha=self.config.FOCAL_ALPHA, 
                                        gamma=self.config.FOCAL_GAMMA)
        else:
            loss = F.binary_cross_entropy_with_logits(bag_logits, bag_labels)
        
        return loss, bag_logits, attention_weights


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    """
    Focal Loss for imbalanced classification (works with logits)
    Safe for autocast/mixed precision training
    """
    BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    
    # Focal term
    focal_term = (1 - pt) ** gamma
    
    # Alpha balancing
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    F_loss = alpha_t * focal_term * BCE_loss
    
    return F_loss.mean()


def focal_loss(inputs, targets, alpha=0.75, gamma=2.0):
    """
    Focal Loss for imbalanced classification (deprecated - use focal_loss_with_logits)
    """
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    
    # Focal term
    focal_term = (1 - pt) ** gamma
    
    # Alpha balancing
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    F_loss = alpha_t * focal_term * BCE_loss
    
    return F_loss.mean()


# ============================================================================
# DATASET
# ============================================================================

class MILDataset(Dataset):
    """
    Dataset for Multiple Instance Learning
    Each sample is a bag containing multiple instances (reads)
    """
    
    def __init__(self, df, labels_df, config, augment=True):
        self.config = config
        self.augment = augment
        
        # Merge with labels
        df_merged = df.merge(
            labels_df[['transcript_id', 'transcript_position', 'label', 'gene_id']],
            on=['transcript_id', 'transcript_position'], 
            how='inner'
        )
        
        self.feature_cols = ['dwell_-1', 'std_-1', 'mean_-1',
                            'dwell_0', 'std_0', 'mean_0',
                            'dwell_+1', 'std_+1', 'mean_+1']
        
        # Group by site (bag)
        self.bags = []
        self.bag_labels = []
        self.gene_ids = []
        
        grouped = df_merged.groupby(['transcript_id', 'transcript_position'])
        
        for (transcript_id, position), site_df in tqdm(grouped, desc="Creating MIL dataset"):
            instances = site_df[self.feature_cols].values
            
            if len(instances) >= config.N_READS_PER_SITE:
                self.bags.append(instances)
                self.bag_labels.append(site_df['label'].iloc[0])
                self.gene_ids.append(site_df['gene_id'].iloc[0])
        
        self.bag_labels = np.array(self.bag_labels, dtype=np.float32)
        self.gene_ids = np.array(self.gene_ids)
        
        # Compute normalization statistics
        all_instances = np.vstack(self.bags)
        self.mean = np.mean(all_instances, axis=0)
        self.std = np.std(all_instances, axis=0) + 1e-8
        
        pos_count = sum(self.bag_labels)
        print(f"MIL Dataset: {len(self.bags)} bags")
        print(f"  Positive bags: {pos_count} ({pos_count/len(self.bags)*100:.1f}%)")
        print(f"  Negative bags: {len(self.bags)-pos_count} ({(len(self.bags)-pos_count)/len(self.bags)*100:.1f}%)")
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        instances = self.bags[idx].copy()
        bag_label = self.bag_labels[idx]
        
        # Sample or pad instances
        n_instances = len(instances)
        if n_instances > self.config.N_READS_PER_SITE:
            indices = np.random.choice(n_instances, self.config.N_READS_PER_SITE, replace=False)
            instances = instances[indices]
        elif n_instances < self.config.N_READS_PER_SITE:
            indices = np.random.choice(n_instances, self.config.N_READS_PER_SITE, replace=True)
            instances = instances[indices]
        
        # Normalize
        instances = (instances - self.mean) / self.std
        
        # Augmentation
        if self.augment and np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.05, instances.shape)
            instances = instances + noise
        
        return torch.FloatTensor(instances), torch.FloatTensor([bag_label])


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, config, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for bag, bag_label in pbar:
        bag = bag.to(device)
        bag_label = bag_label.squeeze(-1).to(device)
        
        optimizer.zero_grad()
        
        if config.MIXED_PRECISION and scaler is not None:
            with autocast():
                loss, bag_logits, _ = model.calculate_objective(bag, bag_label)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, bag_logits, _ = model.calculate_objective(bag, bag_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
        
        # Convert logits to probabilities for metrics
        bag_probs = torch.sigmoid(bag_logits)
        
        total_loss += loss.item()
        all_preds.extend(bag_probs.detach().cpu().numpy())
        all_labels.extend(bag_label.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    
    return avg_loss, auc


def evaluate(model, val_loader, device, config):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attentions = []
    
    with torch.no_grad():
        for bag, bag_label in tqdm(val_loader, desc="Evaluating"):
            bag = bag.to(device)
            bag_label = bag_label.squeeze(-1).to(device)
            
            if config.MIXED_PRECISION:
                with autocast():
                    loss, bag_logits, attention = model.calculate_objective(bag, bag_label)
            else:
                loss, bag_logits, attention = model.calculate_objective(bag, bag_label)
            
            # Convert logits to probabilities
            bag_probs = torch.sigmoid(bag_logits)
            
            total_loss += loss.item()
            all_preds.extend(bag_probs.cpu().numpy())
            all_labels.extend(bag_label.cpu().numpy())
            all_attentions.extend(attention.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    pr_auc = average_precision_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    
    # Find best threshold
    best_f1 = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds_binary = (np.array(all_preds) >= thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return avg_loss, auc, pr_auc, all_preds, all_labels, all_attentions, best_threshold, best_f1


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_mil_model(config):
    """Create MIL model based on config"""
    if config.MIL_TYPE == 'attention':
        return AttentionMIL(config)
    elif config.MIL_TYPE == 'instance':
        return InstanceMIL(config)
    elif config.MIL_TYPE == 'gated_attention':
        return GatedAttentionMIL(config)
    else:
        raise ValueError(f"Unknown MIL type: {config.MIL_TYPE}")


# ============================================================================
# UTILITIES
# ============================================================================

def setup_logging(output_dir):
    """Setup logging"""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(data_file, labels_file):
    """Load dataset"""
    print("Loading dataset...")
    rows = []
    
    with gzip.open(data_file, 'rt', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with gzip.open(data_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Loading data"):
            data = json.loads(line)
            for transcript_id, positions in data.items():
                for transcript_position, sequences in positions.items():
                    for sequence, feature_list in sequences.items():
                        for features in feature_list:
                            rows.append({
                                'transcript_id': transcript_id,
                                'transcript_position': int(transcript_position),
                                'dwell_-1': features[0],
                                'std_-1': features[1],
                                'mean_-1': features[2],
                                'dwell_0': features[3],
                                'std_0': features[4],
                                'mean_0': features[5],
                                'dwell_+1': features[6],
                                'std_+1': features[7],
                                'mean_+1': features[8],
                            })
    
    df = pd.DataFrame(rows)
    labels = pd.read_csv(labels_file)
    
    print(f"Loaded {len(df)} reads from {len(df.groupby(['transcript_id', 'transcript_position']))} sites")
    return df, labels


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    config = Config()
    logger = setup_logging(config.OUTPUT_DIR)
    
    logger.info("="*80)
    logger.info("M6ANET MULTIPLE INSTANCE LEARNING FRAMEWORK")
    logger.info("="*80)
    logger.info(f"\nMIL Type: {config.MIL_TYPE}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    df, labels = load_data(config.DATA_FILE, config.LABELS_FILE)
    full_dataset = MILDataset(df, labels, config)
    
    # Cross-validation
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    fold_results = []
    all_test_preds = []
    all_test_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(
        full_dataset.bags, full_dataset.bag_labels, full_dataset.gene_ids)):
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold + 1}/{config.N_FOLDS}")
        logger.info(f"{'='*80}")
        
        # Create fold datasets
        train_bags = [full_dataset.bags[i] for i in train_idx]
        train_labels = full_dataset.bag_labels[train_idx]
        val_bags = [full_dataset.bags[i] for i in val_idx]
        val_labels = full_dataset.bag_labels[val_idx]
        
        train_dataset = MILDataset.__new__(MILDataset)
        train_dataset.bags = train_bags
        train_dataset.bag_labels = train_labels
        train_dataset.config = config
        train_dataset.augment = True
        train_dataset.mean = full_dataset.mean
        train_dataset.std = full_dataset.std
        
        val_dataset = MILDataset.__new__(MILDataset)
        val_dataset.bags = val_bags
        val_dataset.bag_labels = val_labels
        val_dataset.config = config
        val_dataset.augment = False
        val_dataset.mean = full_dataset.mean
        val_dataset.std = full_dataset.std
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        # Create MIL model
        model = create_mil_model(config).to(device)
        
        if fold == 0:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\nModel: {config.MIL_TYPE.upper()} MIL")
            logger.info(f"Total parameters: {total_params:,}")
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        scaler = GradScaler() if config.MIXED_PRECISION else None
        
        # Training loop
        best_val_pr_auc = 0
        patience_counter = 0
        
        for epoch in range(config.EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{config.EPOCHS}")
            
            train_loss, train_auc = train_epoch(
                model, train_loader, optimizer, device, config, scaler
            )
            val_loss, val_auc, val_pr_auc, val_preds, val_labels, val_attentions, best_thresh, best_f1 = evaluate(
                model, val_loader, device, config
            )
            
            logger.info(
                f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}, "
                f"F1: {best_f1:.4f} @{best_thresh:.3f}"
            )
            
            # Save best model
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                patience_counter = 0
                
                checkpoint_dir = Path(config.CHECKPOINT_DIR)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / f'best_model_fold{fold}.pt')
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(checkpoint_dir / f'best_model_fold{fold}.pt'))
        _, final_auc, final_pr_auc, final_preds, final_labels, final_attentions, final_thresh, final_f1 = evaluate(
            model, val_loader, device, config
        )
        
        fold_results.append({
            'fold': fold + 1,
            'roc_auc': final_auc,
            'pr_auc': final_pr_auc,
            'best_f1': final_f1,
            'best_threshold': final_thresh
        })
        
        all_test_preds.extend(final_preds)
        all_test_labels.extend(final_labels)
        
        logger.info(f"\nFold {fold+1} Final: ROC-AUC={final_auc:.4f}, PR-AUC={final_pr_auc:.4f}, F1={final_f1:.4f}")
    
    # Overall results
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*80}")
    
    mean_roc = np.mean([r['roc_auc'] for r in fold_results])
    std_roc = np.std([r['roc_auc'] for r in fold_results])
    mean_pr = np.mean([r['pr_auc'] for r in fold_results])
    std_pr = np.std([r['pr_auc'] for r in fold_results])
    mean_f1 = np.mean([r['best_f1'] for r in fold_results])
    
    overall_roc = roc_auc_score(all_test_labels, all_test_preds)
    overall_pr = average_precision_score(all_test_labels, all_test_preds)
    
    logger.info(f"\nPer-fold average:")
    logger.info(f"  ROC-AUC: {mean_roc:.4f} ± {std_roc:.4f}")
    logger.info(f"  PR-AUC:  {mean_pr:.4f} ± {std_pr:.4f}")
    logger.info(f"  F1:      {mean_f1:.4f}")
    logger.info(f"\nCombined:")
    logger.info(f"  ROC-AUC: {overall_roc:.4f}")
    logger.info(f"  PR-AUC:  {overall_pr:.4f}")
    
    # Save results
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_test_labels, all_test_preds)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'{config.MIL_TYPE.upper()} MIL (AUC = {overall_roc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {config.MIL_TYPE.upper()} MIL', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot PR curve
    precision, recall, _ = precision_recall_curve(all_test_labels, all_test_preds)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=2, label=f'{config.MIL_TYPE.upper()} MIL (AUC = {overall_pr:.4f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curve - {config.MIL_TYPE.upper()} MIL', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize attention weights for a few samples
    if config.MIL_TYPE in ['attention', 'gated_attention']:
        plot_attention_examples(model, val_dataset, device, output_dir, config)
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_label': all_test_labels,
        'predicted_probability': all_test_preds
    })
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("M6ANET MULTIPLE INSTANCE LEARNING RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"MIL Framework: {config.MIL_TYPE.upper()}\n\n")
        f.write("MIL Characteristics:\n")
        f.write("  - Each site is a 'bag' of reads (instances)\n")
        f.write("  - Bag is positive if at least one read shows modification\n")
        f.write("  - Learns which reads are most informative\n")
        f.write("  - Provides interpretability through attention weights\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Hidden dim: {config.HIDDEN_DIM}\n")
        f.write(f"  Attention dim: {config.ATTENTION_DIM}\n")
        f.write(f"  Dropout: {config.DROPOUT}\n")
        f.write(f"  Focal loss: {config.USE_FOCAL_LOSS}\n")
        if config.USE_FOCAL_LOSS:
            f.write(f"  Focal alpha: {config.FOCAL_ALPHA}\n")
            f.write(f"  Focal gamma: {config.FOCAL_GAMMA}\n")
        f.write(f"\nTraining:\n")
        f.write(f"  Epochs: {config.EPOCHS}\n")
        f.write(f"  Batch size: {config.BATCH_SIZE}\n")
        f.write(f"  Learning rate: {config.LEARNING_RATE}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Combined ROC-AUC: {overall_roc:.4f}\n")
        f.write(f"  Combined PR-AUC: {overall_pr:.4f}\n")
        f.write(f"  Mean ROC-AUC: {mean_roc:.4f} ± {std_roc:.4f}\n")
        f.write(f"  Mean PR-AUC: {mean_pr:.4f} ± {std_pr:.4f}\n")
        f.write(f"  Mean F1: {mean_f1:.4f}\n\n")
        f.write(f"Per-fold results:\n")
        for r in fold_results:
            f.write(f"  Fold {r['fold']}: ROC-AUC={r['roc_auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, "
                   f"F1={r['best_f1']:.4f} @thresh={r['best_threshold']:.3f}\n")
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("Training complete!")


def plot_attention_examples(model, dataset, device, output_dir, config, n_examples=5):
    """
    Visualize attention weights for example bags
    Shows which reads the model pays attention to
    """
    model.eval()
    
    # Select examples (mix of positive and negative)
    pos_indices = np.where(dataset.bag_labels == 1)[0]
    neg_indices = np.where(dataset.bag_labels == 0)[0]
    
    selected_indices = list(np.random.choice(pos_indices, min(3, len(pos_indices)), replace=False))
    selected_indices += list(np.random.choice(neg_indices, min(2, len(neg_indices)), replace=False))
    
    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(12, 3*len(selected_indices)))
    if len(selected_indices) == 1:
        axes = [axes]
    
    with torch.no_grad():
        for idx, bag_idx in enumerate(selected_indices):
            bag, bag_label = dataset[bag_idx]
            bag = bag.unsqueeze(0).to(device)
            
            _, attention_weights = model(bag)
            attention_weights = attention_weights.cpu().numpy()[0]
            
            # Plot attention weights
            ax = axes[idx]
            reads = np.arange(len(attention_weights))
            colors = ['red' if bag_label.item() == 1 else 'blue']
            
            ax.bar(reads, attention_weights, color=colors[0], alpha=0.6)
            ax.set_xlabel('Read Index', fontsize=12)
            ax.set_ylabel('Attention Weight', fontsize=12)
            ax.set_title(f'Bag {bag_idx} - Label: {"Positive" if bag_label.item() == 1 else "Negative"}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Highlight top attended reads
            top_k = 3
            top_indices = np.argsort(attention_weights)[-top_k:]
            for top_idx in top_indices:
                ax.axvline(x=top_idx, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention visualization saved to {output_dir / 'attention_examples.png'}")


if __name__ == '__main__':
    main()