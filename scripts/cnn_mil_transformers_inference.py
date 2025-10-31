# # CNN-Transformer-MIL Inference
# 
# This notebook reproduces the preprocessing and model architecture used for training, loads a saved checkpoint, and runs inference on unlabeled data to produce per-site m6A probabilities.
# 
# Usage notes:
# - Edit the parameters below before running if using a different dataset.
# - The notebook will try to reuse a saved scaler if provided.
# - The model architecture is identical to training so that state dicts load cleanly.



# Imports
import gzip
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')



# Parameters (edit before running)
DATA = 'dataset2' # dataset name
DATA_FILE = f'../data/{DATA}.json.gz'
MODEL_PATH = '../models/final_model_state_dict.pt'
SCALER_PATH = '../models/' 
OUTPUT_CSV = f'../predictions/{DATA}_predictions_transformers.csv'
BATCH_SIZE = 64
BAG_SIZE = 40
VERBOSE = True



# Load raw JSON.gz and build DataFrame of reads
def load_unlabeled_data(data_file):
    rows = []
    with gzip.open(data_file, 'rt', encoding='utf-8') as f:
        total = sum(1 for _ in f)
    with gzip.open(data_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc='Loading raw data'):
            data = json.loads(line)
            for transcript_id, positions in data.items():
                for transcript_position, sequences in positions.items():
                    for sequence, feature_list in sequences.items():
                        for features in feature_list:
                            rows.append({
                                'transcript_id': transcript_id,
                                'transcript_position': int(transcript_position),
                                'sequence': sequence,
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
    # Add derived features that are the same as training
    df['mean_0_minus_mean_-1'] = df['mean_0'] - df['mean_-1']
    df['mean_0_minus_mean_+1'] = df['mean_0'] - df['mean_+1']
    df['dwell_0_minus_dwell_-1'] = df['dwell_0'] - df['dwell_-1']
    df['dwell_0_minus_dwell_+1'] = df['dwell_0'] - df['dwell_+1']
    df['std_0_minus_avg_neighbor_std'] = df['std_0'] - ((df['std_-1'] + df['std_+1']) / 2.0)
    if VERBOSE:
        # Use string column names for grouping to avoid KeyError when variables of same names exist
        n_sites = df.groupby(['transcript_id', 'transcript_position']).ngroups if not df.empty else 0
        print(f'Loaded {len(df)} reads from {n_sites} sites')
    return df

# Run loading of data
df_raw = load_unlabeled_data(DATA_FILE)
df_raw.head()



# Encoding helpers and numeric feature list
BASE2IDX = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
PAD_IDX = 4

def seq_to_idx7(s: str):
    s = str(s).upper().replace('U', 'T')
    return np.array([BASE2IDX.get(ch, 0) for ch in s], dtype=np.int64)

# numeric columns used by model
num_cols = [
    'dwell_-1','std_-1','mean_-1',
    'dwell_0','std_0','mean_0',
    'dwell_+1','std_+1','mean_+1',
    'mean_0_minus_mean_-1',
    'mean_0_minus_mean_+1',
    'dwell_0_minus_dwell_-1',
    'dwell_0_minus_dwell_+1',
    'std_0_minus_avg_neighbor_std'
]
seq_col = 'sequence'
site_key = ['transcript_id', 'transcript_position']



# Create bags for the data (MIL approach)
def create_bags_unlabeled(df, site_key, num_cols, seq_col, min_reads=1, max_reads=50):
    bags = []
    grouped = df.groupby(site_key)
    for site, group in grouped:
        if len(group) < min_reads:
            continue
        features = group[num_cols].to_numpy(dtype=np.float32)
        sequences = group[seq_col].astype(str).tolist()
        seq_idx = np.vstack([seq_to_idx7(s) for s in sequences])
        n = len(features)
        if n > max_reads:
            idx = np.random.choice(n, max_reads, replace=False)
            features = features[idx]
            seq_idx = seq_idx[idx]
        bags.append({
            'site': site,
            'transcript_id': site[0],
            'transcript_position': int(site[1]),
            'features': features,
            'seq_idx': seq_idx,
            'n_reads': len(features),
            'label': 0,  # placeholder
            'gene_id': ''
        })
    print(f'Created {len(bags)} bags from {len(grouped)} sites')
    return bags

# Build bags from raw dataframe
bags = create_bags_unlabeled(df_raw, site_key, num_cols, seq_col, min_reads=1, max_reads=BAG_SIZE)



# Reuse RNA_MIL_Dataset from training
class RNA_MIL_Dataset_Unlabeled(Dataset):
    def __init__(self, bags, bag_size=40, pad_idx=PAD_IDX):
        self.proc = []
        self.meta = []
        for bag in bags:
            num = bag['features']
            seq = bag['seq_idx']
            n = bag['n_reads']
            if n == 0:
                continue
            if n < bag_size:
                pad_num = np.zeros((bag_size - n, num.shape[1]), dtype=np.float32)
                pad_seq = np.full((bag_size - n, seq.shape[1]), pad_idx, dtype=np.int64)
                num_fixed = np.vstack([num, pad_num])
                seq_fixed = np.vstack([seq, pad_seq])
                mask = np.zeros(bag_size, dtype=np.float32)
                mask[:n] = 1.0
            else:
                if n > bag_size:
                    idx = np.arange(n)[:bag_size]
                    num_fixed = num[idx].astype(np.float32)
                    seq_fixed = seq[idx].astype(np.int64)
                    mask = np.ones(bag_size, dtype=np.float32)
                else:
                    num_fixed = num.astype(np.float32)
                    seq_fixed = seq.astype(np.int64)
                    mask = np.ones(bag_size, dtype=np.float32)
            self.proc.append({'num': num_fixed, 'seq': seq_fixed, 'mask': mask})
            self.meta.append({'transcript_id': bag['transcript_id'], 'transcript_position': bag['transcript_position'], 'n_reads': bag['n_reads']})
    def __len__(self):
        return len(self.proc)
    def __getitem__(self, idx):
        b = self.proc[idx]
        x_num = torch.from_numpy(b['num'])
        x_seq = torch.from_numpy(b['seq'])
        mask  = torch.from_numpy(b['mask'])
        return x_num, x_seq, mask
    def get_metadata(self, idx):
        return self.meta[idx]

# Create dataset and dataloader
dataset = RNA_MIL_Dataset_Unlabeled(bags, bag_size=BAG_SIZE, pad_idx=PAD_IDX)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
print(f'Dataset size: {len(dataset)} bags, DataLoader batches: {len(dataloader)}')



# Initialise Model classes from training
class SeqEmbCNN(nn.Module):
    def __init__(self, vocab=5, d_emb=8, kernel_sizes=(2,3,4,5), n_filters=32, d_out=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_emb, n_filters, ks, padding=0) for ks in kernel_sizes
        ])
        self.attentions = nn.ModuleList([
            nn.Linear(n_filters, 1) for _ in kernel_sizes
        ])
        self.proj = nn.Linear(n_filters * len(kernel_sizes), d_out)
        self.norm = nn.LayerNorm(d_out)
    def forward(self, x_idx):
        X = self.emb(x_idx).transpose(1,2)
        feats = []
        for conv, att in zip(self.convs, self.attentions):
            h = torch.nn.functional.gelu(conv(X))
            h_t = h.permute(0,2,1)
            scores = att(h_t).squeeze(-1)
            weights = torch.nn.functional.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (h_t * weights).sum(dim=1)
            feats.append(pooled)
        z = torch.cat(feats, dim=1)
        z = self.proj(z)
        return self.norm(torch.nn.functional.gelu(z))

class GatedAttentionPooling(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Dropout(dropout), nn.Linear(d_model,1)) for _ in range(n_heads)
        ])
        self.gate = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2,1), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(d_model * n_heads, d_model), nn.LayerNorm(d_model), nn.GELU())
    def forward(self, h, mask):
        gates = self.gate(h).squeeze(-1) * mask
        pooled = []
        all_weights = []
        for attn in self.attention_heads:
            scores = attn(h).squeeze(-1)
            gated_scores = scores * gates
            gated_scores = gated_scores.masked_fill(mask == 0, float('-inf'))
            weights = torch.nn.functional.softmax(gated_scores, dim=1).unsqueeze(-1)
            pooled.append((h * weights).sum(dim=1))
            all_weights.append(weights.squeeze(-1))
        bag_repr = self.fusion(torch.cat(pooled, dim=-1))
        avg_weights = torch.stack(all_weights, dim=0).mean(dim=0)
        return bag_repr, avg_weights, gates

class TransformerMIL(nn.Module):
    def __init__(self, num_features=9, d_model=128, n_heads=4, n_layers=4, d_ff=1024, dropout=0.1, attn_pool_heads=4, instance_dropout=0.15):
        super().__init__()
        self.instance_dropout = instance_dropout
        self.seq_encoder = SeqEmbCNN(vocab=5, d_emb=8, kernel_sizes=(2,3,4,5), n_filters=32, d_out=64)
        self.num_proj = nn.Sequential(nn.Linear(num_features, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout))
        self.feature_fusion = nn.Sequential(nn.Linear(128, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attention_pool = GatedAttentionPooling(d_model, n_heads=attn_pool_heads, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model//2), nn.LayerNorm(d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, d_model//4), nn.LayerNorm(d_model//4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//4, 1))
        self.instance_classifier = nn.Sequential(nn.Linear(d_model, d_model//2), nn.LayerNorm(d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, d_model//4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//4, 1))
    def encode_sequences(self, x_seq):
        B, K, L = x_seq.shape
        seq_flat = x_seq.reshape(B * K, L)
        z_seq = self.seq_encoder(seq_flat)
        return z_seq.view(B, K, -1)
    def apply_instance_dropout(self, mask):
        if not self.training or self.instance_dropout == 0:
            return mask
        B, K = mask.shape
        dropout_mask = torch.rand(B, K, device=mask.device) > self.instance_dropout
        n_keep = (dropout_mask * mask).sum(dim=1, keepdim=True)
        too_few = n_keep < 5
        dropout_mask = torch.where(too_few, torch.ones_like(dropout_mask), dropout_mask)
        return mask * dropout_mask.float()
    def forward(self, x_num, mask, x_seq=None):
        B, K, _ = x_num.shape
        effective_mask = self.apply_instance_dropout(mask)
        num_features = self.num_proj(x_num)
        if x_seq is not None:
            seq_features = self.encode_sequences(x_seq)
            combined = torch.cat([num_features, seq_features], dim=-1)
        else:
            combined = num_features
        h = self.feature_fusion(combined)
        src_key_padding_mask = (effective_mask == 0)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        instance_logits = self.instance_classifier(h).squeeze(-1)
        instance_logits = instance_logits.masked_fill(effective_mask == 0, float('-inf'))
        bag_repr, attention_weights, gates = self.attention_pool(h, effective_mask)
        bag_logits = self.classifier(bag_repr).squeeze(-1)
        return bag_logits, attention_weights, instance_logits, gates



# Instantiate model and load weights
model = TransformerMIL(num_features=len(num_cols))
device = torch.device(DEVICE)
model = model.to(device)

# Try loading different checkpoint formats
if MODEL_PATH is not None and Path(MODEL_PATH).exists():
    ckpt = torch.load(MODEL_PATH, map_location=device)
    # If it's a plain state_dict (mapping of tensors), load directly
    if isinstance(ckpt, dict) and any(k.startswith('model') or k.endswith('state_dict') for k in ckpt.keys()):
        # support different key names
        if 'model_state' in ckpt:
            state = ckpt['model_state']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
        else:
            # assume ckpt itself is state_dict-like
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state)
    print(f'Loaded model weights from {MODEL_PATH}')
else:
    print(f'MODEL_PATH does not exist: {MODEL_PATH}. Please set MODEL_PATH to a valid checkpoint.')



# Scaler loading
scaler_path = 'scaler.joblib'
if SCALER_PATH:
    p = Path(SCALER_PATH)
    scaler_path = p if p.is_file() else (p / 'scaler.joblib')
elif MODEL_PATH:
    scaler_path = Path(MODEL_PATH).parent / 'scaler.joblib'

amp_path = 'amp_grad_scaler.pt'
if SCALER_PATH:
    p = Path(SCALER_PATH)
    amp_path = (p / 'amp_grad_scaler.pt') if p.is_dir() else (p if p.name == 'amp_grad_scaler.pt' else None)
elif MODEL_PATH:
    amp_path = Path(MODEL_PATH).parent / 'amp_grad_scaler.pt'

# Load feature scaler if scaler_path exists
feature_scaler = None
saved_num_cols = None
if scaler_path is not None and Path(scaler_path).exists():
    payload = joblib.load(scaler_path)
    if isinstance(payload, dict) and 'scaler' in payload:
        feature_scaler = payload['scaler']
        saved_num_cols = payload.get('num_cols', None)
    else:
        feature_scaler = payload
    print(f'Loaded feature scaler from {scaler_path}')
else:
    print('No scaler_path found; feature_scaler remains None')

# Load AMP GradScaler directly if amp_path exists (minimal loading)
amp_scaler = None
if amp_path is not None and Path(amp_path).exists():
    amp_state = torch.load(amp_path, map_location='cpu')
    amp_scaler = GradScaler()
    amp_scaler.load_state_dict(amp_state)
    print(f'Loaded AMP GradScaler from {amp_path}')
else:
    print('No amp_path found; amp_scaler remains None')

# Verify column consistency when possible
if feature_scaler is not None and saved_num_cols is not None:
    if list(saved_num_cols) != list(num_cols):
        print('WARNING: numeric column list in saved scaler differs from current num_cols.')
        print(f'Saved: {saved_num_cols}')
        print(f'Current: {num_cols}')
    else:
        print('Saved scaler column order matches current num_cols')

# Apply scaling if we have a feature scaler
if feature_scaler is not None:
    df_raw[num_cols] = feature_scaler.transform(df_raw[num_cols])
    print('Applied feature scaler to df_raw')

# Rebuild bags and dataset now that df_raw is scaled
bags = create_bags_unlabeled(df_raw, site_key, num_cols, seq_col, min_reads=1, max_reads=BAG_SIZE)
dataset = RNA_MIL_Dataset_Unlabeled(bags, bag_size=BAG_SIZE, pad_idx=PAD_IDX)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
print(f'Dataset size: {len(dataset)} bags, DataLoader batches: {len(dataloader)}')



# Run inference and save per-site probabilities
model.eval()
results = []
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Running inference')):
        x_num, x_seq, mask = batch
        x_num = x_num.to(device)
        x_seq = x_seq.to(device)
        mask  = mask.to(device)
        bag_logits, attn, inst_logits, gates = model(x_num=x_num, x_seq=x_seq, mask=mask)
        probs = torch.sigmoid(bag_logits).cpu().numpy()
        batch_start = batch_idx * BATCH_SIZE
        for i in range(len(probs)):
            meta = dataset.get_metadata(batch_start + i)
            results.append({
                'transcript_id': meta['transcript_id'],
                'transcript_position': meta['transcript_position'],
                'score': float(probs[i])
            })

# Build DataFrame and save results
pred_df = pd.DataFrame(results)
out_path = Path(OUTPUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)
pred_df.to_csv(out_path, index=False)
print(f'Saved predictions to {out_path}')
pred_df.head()

