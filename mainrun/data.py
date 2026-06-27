"""
    Dataset loading, tokenisation, and batch generation 
    utilities.
"""

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


## Loads num_titles HN headlines, shuffles with seed, and splits into train/val lists.
def get_titles(num_titles: int, seed: int, val_frac: float) -> tuple[list[str], list[str]]:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]


## Returns one (x, y) batch from split_ids at ptr, resetting ptr to 0 if near the end.
def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size


## Yields every non-overlapping (x, y) batch from split_ids, stopping before any remainder.
def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y


## Trains a byte-level BPE tokenizer on titles and returns the fitted Tokenizer.
def train_tokenizer(
    titles: list[str],
    vocab_size: int,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token],
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


## Thin wrapper around a HuggingFace Tokenizer exposing encode, decode, and vocab_size.
class BPETokenizer:
    ### Stores the underlying HuggingFace tokenizer instance.
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer

    ### Returns the token ID sequence for the input string.
    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    ### Reconstructs a string from token IDs, stripping special tokens.
    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    ### Returns the total number of tokens in the vocabulary.
    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()
