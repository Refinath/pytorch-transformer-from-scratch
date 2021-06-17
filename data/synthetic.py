"""
Utilities to generate synthetic sequence‑to‑sequence datasets.

The toy dataset used in this project is a simple copy task: given a random
sequence of integers, the model must reproduce the same sequence shifted by
one position.  This synthetic problem allows us to test whether the
Transformer can learn to attend to the appropriate positions without the
overhead of tokenisation or external corpora.  Padding tokens are used to
allow variable length sequences within a batch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticSequenceDataset(Dataset):
    """Dataset generating random integer sequences for a copy task.

    Each sample consists of a source sequence of integers and a target sequence
    obtained by shifting the source one position to the right and prepending a
    special beginning‑of‑sequence token.  The sequences are padded to a fixed
    maximum length using the PAD token (0).

    Args:
        num_samples: Number of samples in the dataset.
        max_len: Maximum length of sequences (including padding).
        min_len: Minimum actual length of sequences (without padding).
        vocab_size: Size of the vocabulary (including PAD, BOS and EOS tokens).
        pad_id: Index of the padding token.
        bos_id: Index of the beginning‑of‑sequence token.
        eos_id: Index of the end‑of‑sequence token.
    """

    num_samples: int
    max_len: int
    min_len: int
    vocab_size: int
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def __len__(self) -> int:
        return self.num_samples

    def _random_sequence(self) -> List[int]:
        # Exclude special tokens from the random range
        length = random.randint(self.min_len, self.max_len - 2)  # leave space for EOS
        seq = [random.randint(self.eos_id + 1, self.vocab_size - 1) for _ in range(length)]
        seq.append(self.eos_id)
        return seq

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random sequence
        src_seq = self._random_sequence()
        # Target input starts with BOS and then the src sequence (without EOS)
        tgt_in = [self.bos_id] + src_seq[:-1]
        # Pad sequences
        src_padded = src_seq + [self.pad_id] * (self.max_len - len(src_seq))
        tgt_padded = tgt_in + [self.pad_id] * (self.max_len - len(tgt_in))
        return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)
