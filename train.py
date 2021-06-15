"""
Training script for the Transformer model on a synthetic copy task.

This script generates a toy dataset using `SyntheticSequenceDataset` and trains
a Transformer model to reproduce the input sequence.  It logs training and
validation loss at each epoch and saves the trained model to a checkpoint file
and the log to a CSV file.  You can customise hyperparameters such as model
size, number of layers, learning rate and dataset size via command‑line
arguments.

Example usage:

    python train.py --epochs 20 --batch-size 32 --d-model 128 --num-layers 2

References:
* The model uses cross‑entropy loss with an `ignore_index` to ignore padding
  tokens during training【565121574583199†L2973-L3040】.
* The target mask prevents attending to future positions, ensuring the
  auto‑regressive property【751605239765780†L296-L305】.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from data.synthetic import SyntheticSequenceDataset
from model import Transformer


def collate_fn(batch: list[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function to stack source and target sequences into tensors."""
    src_batch, tgt_batch = zip(*batch)
    src_tensor = torch.stack(src_batch)
    tgt_tensor = torch.stack(tgt_batch)
    return src_tensor, tgt_tensor


def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss per token."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt_in in dataloader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        # Target output is shifted right by one position
        tgt_out = src.clone().to(device)  # For copy task, target output is src sequence
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        # Reshape for loss: (batch*tgt_len, vocab)
        loss = criterion(
            logits.view(-1, model.tgt_vocab_size), tgt_out.view(-1)
        )
        loss.backward()
        optimizer.step()
        # Compute non‑pad tokens
        num_tokens = (tgt_out != model.pad_id).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    return total_loss / total_tokens


def evaluate_epoch(
    model: Transformer,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt_in in dataloader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = src.clone().to(device)
            logits = model(src, tgt_in)
            loss = criterion(
                logits.view(-1, model.tgt_vocab_size), tgt_out.view(-1)
            )
            num_tokens = (tgt_out != model.pad_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    return total_loss / total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer from scratch on a synthetic copy task")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset-size", type=int, default=1000, help="Total number of samples in the dataset")
    parser.add_argument("--train-val-split", type=float, default=0.8, help="Fraction of data used for training")
    parser.add_argument("--max-len", type=int, default=20, help="Maximum sequence length")
    parser.add_argument("--min-len", type=int, default=5, help="Minimum sequence length")
    parser.add_argument("--vocab-size", type=int, default=50, help="Vocabulary size including special tokens")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of encoder/decoder layers")
    parser.add_argument("--d-ff", type=int, default=256, help="Feed‑forward network hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--logfile", type=str, default="training_log.csv", help="CSV file to write training log")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to save model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and split
    dataset = SyntheticSequenceDataset(
        num_samples=args.dataset_size,
        max_len=args.max_len,
        min_len=args.min_len,
        vocab_size=args.vocab_size,
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    train_size = int(args.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Instantiate model
    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_len,
        dropout=args.dropout,
        pad_id=0,
    )
    model.to(device)

    # Loss and optimiser
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore padding【565121574583199†L2973-L3040】
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # Logging
    log_path = Path(args.logfile)
    with log_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_epoch(model, val_loader, criterion, device)
            writer.writerow([epoch, train_loss, val_loss])
            print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    # Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, args.checkpoint)
    print(f"Model saved to {args.checkpoint} and log written to {args.logfile}")


if __name__ == "__main__":
    main()
