"""Transformer training entrypoint.

This script expects preprocessed NumPy arrays produced by `preprocess.py`:
- X.npy (N, L) int pitches
- y.npy (N,) next pitch labels
Optional aligned conditioning arrays:
- X_vel.npy (N, L) velocities (0-127 int)
- y_vel.npy (N,) velocity target
- X_dur.npy (N, L) durations (float seconds)
- y_dur.npy (N,) duration target

Usage (example):
  python train.py --data-dir . --batch-size 64 --epochs 30 --device cuda

The script will perform a train/val split if no test set files are present. It saves checkpoints and prints basic metrics.
"""

import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

class MidiDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train"):
        data_dir = Path(data_dir)
        X_path = data_dir / ("X.npy" if split == "train" else "X_test.npy")
        y_path = data_dir / ("y.npy" if split == "train" else "y_test.npy")     
        vel_path = data_dir / ("X_vel.npy" if split == "train" else "X_vel_test.npy")
        dur_path = data_dir / ("X_dur.npy" if split == "train" else "X_dur_test.npy")
        y_vel_path = data_dir / ("y_vel.npy" if split == "train" else "y_vel_test.npy")
        y_dur_path = data_dir / ("y_dur.npy" if split == "train" else "y_dur_test.npy")

        self.X = torch.from_numpy(np.load(X_path)).long()
        self.y = torch.from_numpy(np.load(y_path)).long()
        self.X_vel = torch.from_numpy(np.load(vel_path)).float()
        self.X_dur = torch.from_numpy(np.load(dur_path)).float()
        self.y_vel = torch.from_numpy(np.load(y_vel_path)).float() 
        self.y_dur = torch.from_numpy(np.load(y_dur_path)).float()

        # infer vocab
        self.vocab_size = int(self.X.max().item() + 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_vel = self.X_vel[idx] if self.X_vel is not None else torch.zeros_like(self.X[idx], dtype=torch.float32)
        x_dur = self.X_dur[idx] if self.X_dur is not None else torch.zeros_like(self.X[idx], dtype=torch.float32)
        y_vel = self.y_vel[idx] if self.y_vel is not None else None
        y_dur = self.y_dur[idx] if self.y_dur is not None else None
        return self.X[idx], x_vel, x_dur, self.y[idx], y_vel, y_dur


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerNextNoteModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, nlayers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.vel_proj = nn.Linear(1, d_model)
        self.dur_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.pitch_head = nn.Linear(d_model, vocab_size)
        self.vel_head = nn.Linear(d_model, 1)
        self.dur_head = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, tokens, velocities=None, durations=None):
        b, seq_len = tokens.shape
        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        if velocities is not None:
            v = velocities.unsqueeze(-1)
            x = x + self.vel_proj(v)
        if durations is not None:
            d = durations.unsqueeze(-1)
            x = x + self.dur_proj(d)
        x = self.pos_enc(x)
        x = self.transformer(x)
        last = x[:, -1, :]
        pitch_logits = self.pitch_head(last)
        vel_pred = self.vel_head(last).squeeze(-1)
        dur_pred = self.dur_head(last).squeeze(-1)
        return pitch_logits, vel_pred, dur_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='Training Data', help='Directory containing preprocessed arrays (X.npy/y.npy) or preprocessed training data (defaults to "Training Data")')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30) # DEFAULT = 30
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--nlayers', type=int, default=4)
    p.add_argument('--ffn-dim', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seq-len', type=int, default=20)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-path', type=str, default='checkpoint.pth')
    p.add_argument('--save-every', type=int, default=1)
    return p.parse_args()


def collate_fn(batch):
    # batch: list of tuples (x, x_vel, x_dur, y, y_vel, y_dur)
    xs = torch.stack([b[0] for b in batch])
    xvel = torch.stack([b[1] for b in batch])
    xdur = torch.stack([b[2] for b in batch])
    ys = torch.stack([b[3] for b in batch])
    # y_vel/y_dur may be None in items; return None if not present
    yv = None
    yd = None
    if batch[0][4] is not None:
        yv = torch.tensor([b[4] for b in batch]).float()
    if batch[0][5] is not None:
        yd = torch.tensor([b[5] for b in batch]).float()
    return xs, xvel, xdur, ys, yv, yd


def train_main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset (single X.npy/y.npy expected) and split indices if no separate test files
    ds_full = MidiDataset(data_dir, split='train')
    N = len(ds_full)
    print(f'Loaded dataset with {N} sequences, vocab_size={ds_full.vocab_size}')

    # If explicit test file exists, the user can pass separate data_dir with test files or use preprocess to produce X_test.npy
    has_explicit_test = (data_dir / 'X_test.npy').exists() and (data_dir / 'y_test.npy').exists()

    if has_explicit_test:
        print('Found explicit test files (X_test.npy/y_test.npy). Using them as test set.')
        ds_train = MidiDataset(data_dir, split='train')
        ds_test = MidiDataset(data_dir, split='test')
        # create loader for train and test (no val)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = None
        test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        vocab_size = ds_train.vocab_size
    else:
        # split full dataset into train/val/test by indices (train/val/test fractions: 1 - val_frac - val_frac, val_frac, val_frac)
        val_frac = args.val_frac
        test_frac = val_frac
        indices = np.arange(N)
        np.random.shuffle(indices)
        n_val = max(1, int(round(N * val_frac)))
        n_test = max(1, int(round(N * test_frac)))
        test_idx = indices[:n_test].tolist()
        val_idx = indices[n_test:n_test + n_val].tolist()
        train_idx = indices[n_test + n_val:].tolist()

        train_loader = DataLoader(Subset(ds_full, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(ds_full, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(Subset(ds_full, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        vocab_size = ds_full.vocab_size

    # Build model
    model = TransformerNextNoteModel(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, dim_feedforward=args.ffn_dim, dropout=args.dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        t0 = time.time()
        for X, X_vel, X_dur, y, y_vel, y_dur in train_loader:
            X = X.to(device)
            X_vel = (X_vel.to(device) / 127.0) if X_vel is not None else None
            X_dur = X_dur.to(device) if X_dur is not None else None
            y = y.to(device)

            optimizer.zero_grad()
            logits, vel_pred, dur_pred = model(X, velocities=X_vel, durations=X_dur)
            loss_pitch = ce_loss(logits, y)
            loss = loss_pitch
            # If y_vel/y_dur provided in dataset, include MSE losses (they must be normalized similarly)
            if y_vel is not None:
                yv = y_vel.to(device) / 127.0
                loss_vel = mse_loss(vel_pred, yv)
                loss = loss + 0.01 * loss_vel
            if y_dur is not None:
                yd = y_dur.to(device)
                loss_dur = mse_loss(dur_pred, yd)
                loss = loss + 0.01 * loss_dur

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = X.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        avg_train_loss = total_loss / total_count
        t1 = time.time()
        print(f'Epoch {epoch}/{args.epochs} — train_loss: {avg_train_loss:.4f} — time: {t1-t0:.1f}s')

        # validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            correct = 0
            with torch.no_grad():
                for X, X_vel, X_dur, y, y_vel, y_dur in val_loader:
                    X = X.to(device)
                    X_vel = (X_vel.to(device) / 127.0) if X_vel is not None else None
                    X_dur = X_dur.to(device) if X_dur is not None else None
                    y = y.to(device)
                    logits, vel_pred, dur_pred = model(X, velocities=X_vel, durations=X_dur)
                    loss_pitch = ce_loss(logits, y)
                    loss_batch = loss_pitch
                    if y_vel is not None:
                        yv = y_vel.to(device) / 127.0
                        loss_batch = loss_batch + 0.01 * mse_loss(vel_pred, yv)
                    if y_dur is not None:
                        yd = y_dur.to(device)
                        loss_batch = loss_batch + 0.01 * mse_loss(dur_pred, yd)
                    val_loss += loss_batch.item() * X.size(0)
                    val_count += X.size(0)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y).sum().item()
            avg_val_loss = val_loss / val_count
            acc = correct / val_count
            print(f'  Val_loss: {avg_val_loss:.4f}  Val_acc: {acc:.4f}')
            # save best
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                # include hyperparameters so inference can reconstruct the model
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'hparams': {
                        'vocab_size': int(vocab_size),
                        'd_model': args.d_model,
                        'nhead': args.nhead,
                        'nlayers': args.nlayers,
                        'ffn_dim': args.ffn_dim,
                        'dropout': args.dropout,
                        'seq_len': args.seq_len,
                    }
                }, args.save_path)
                print(f'  Saved best checkpoint to {args.save_path}')

        # optionally save every few epochs
        if epoch % args.save_every == 0 and val_loader is None:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'hparams': {
                    'vocab_size': int(vocab_size),
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'nlayers': args.nlayers,
                    'ffn_dim': args.ffn_dim,
                    'dropout': args.dropout,
                    'seq_len': args.seq_len,
                }
            }, args.save_path)
            print(f'  Saved checkpoint to {args.save_path}')

    # final test evaluation (if available)
    if test_loader is not None:
        model.eval()
        test_loss = 0.0
        test_count = 0
        correct = 0
        with torch.no_grad():
            for X, X_vel, X_dur, y, y_vel, y_dur in test_loader:
                X = X.to(device)
                X_vel = (X_vel.to(device) / 127.0) if X_vel is not None else None
                X_dur = X_dur.to(device) if X_dur is not None else None
                y = y.to(device)
                logits, vel_pred, dur_pred = model(X, velocities=X_vel, durations=X_dur)
                loss_pitch = ce_loss(logits, y)
                loss_batch = loss_pitch
                if y_vel is not None:
                    yv = y_vel.to(device) / 127.0
                    loss_batch = loss_batch + 0.01 * mse_loss(vel_pred, yv)
                if y_dur is not None:
                    yd = y_dur.to(device)
                    loss_batch = loss_batch + 0.01 * mse_loss(dur_pred, yd)
                test_loss += loss_batch.item() * X.size(0)
                test_count += X.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
        avg_test_loss = test_loss / test_count
        acc = correct / test_count
        print(f'Final Test — loss: {avg_test_loss:.4f} acc: {acc:.4f}')


if __name__ == '__main__':
    train_main()