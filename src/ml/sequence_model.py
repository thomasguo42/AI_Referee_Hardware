"""Sequence model training for saber referee."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

from sequence_dataset import (
    PhraseSequenceDataset,
    build_sequence_entries,
    max_sequence_length,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask):
        # mask unused for GRU but kept for future attention models
        out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


def collate_batch(batch):
    sequences = torch.stack([item["sequence"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch]).float()
    folders = [item["folder"] for item in batch]
    sessions = [item["session"] for item in batch]
    return sequences, masks, labels, folders, sessions


@dataclass
class FoldResult:
    sessions: List[str]
    accuracy: float


def train_one_fold(model, train_loader, val_loader, epochs: int = 40):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for sequences, masks, labels, *_ in train_loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(sequences, masks.to(DEVICE))
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for sequences, masks, labels, *_ in val_loader:
            sequences = sequences.to(DEVICE)
            logits = model(sequences, masks.to(DEVICE))
            probs = torch.sigmoid(logits)
            preds.extend((probs.cpu().numpy() >= 0.5).astype(int).tolist())
            gts.extend(labels.numpy().astype(int).tolist())
    accuracy = (np.array(preds) == np.array(gts)).mean()
    return accuracy


def run_sequence_model(root_dir: str = "/workspace/training_data") -> List[FoldResult]:
    entries = build_sequence_entries(root_dir)
    max_len = max_sequence_length(entries)
    dataset = PhraseSequenceDataset(entries, max_len=max_len)

    sessions = np.array([entry.session for entry in entries])
    groups = sessions
    fold_results: List[FoldResult] = []

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train_idx, test_idx in gkf.split(np.arange(len(entries)), np.zeros(len(entries)), groups):
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(test_subset, batch_size=16, shuffle=False, collate_fn=collate_batch)

        input_dim = dataset.feature_dim
        model = GRUClassifier(input_dim).to(DEVICE)
        accuracy = train_one_fold(model, train_loader, test_loader)
        fold_results.append(FoldResult(
            sessions=list(np.unique(sessions[test_idx])),
            accuracy=accuracy,
        ))
    return fold_results


if __name__ == "__main__":
    results = run_sequence_model()
    for fold in results:
        session_str = ','.join(fold.sessions)
        print(f"Fold [{session_str}] accuracy={fold.accuracy:.3f}")
    mean_acc = np.mean([f.accuracy for f in results])
    print(f"Mean accuracy: {mean_acc:.3f}")
