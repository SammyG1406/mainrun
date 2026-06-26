import utils  # side-effect: devcontainer check
import random, time, os
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

from config import Hyperparameters, GPTConfig, MODE
from model import GPT
from data import get_titles, get_batch, iter_full_split, train_tokenizer, BPETokenizer

logger = None

def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_interop_threads(1)
    torch.set_num_threads(os.cpu_count())

    global logger
    logger = utils.configure_logging(args.log_file)

    logger.log("hyperparameters_configured", mode=MODE, **vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"

    # Cache the tokenizer per-tier (vocab depends on num_titles, so MODE=smoke,
    # validate, and full each get their own cached tokenizer file, and the real
    # MODE=full run is always retrained fresh against the full 100k titles).
    tok_cache_path = Path(f"./data/tokenizer_{MODE}.json")
    if tok_cache_path.exists():
        tok = BPETokenizer(Tokenizer.from_file(str(tok_cache_path)))
        logger.log("tokenizer_loaded_from_cache", path=str(tok_cache_path))
    else:
        raw_tok = train_tokenizer(train_titles + val_titles, args.vocab_size, eos_token=eos_token)
        tok_cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw_tok.save(str(tok_cache_path))
        tok = BPETokenizer(raw_tok)
        logger.log("tokenizer_trained_and_cached", path=str(tok_cache_path))

    train_text = eos_token.join(train_titles) + eos_token
    val_text   = eos_token.join(val_titles)   + eos_token
    train_ids  = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids    = torch.tensor(tok.encode(val_text),   dtype=torch.long)

    batches       = len(train_ids) // (args.block_size * args.batch_size)
    max_steps     = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    model = torch.compile(model)

    decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    warmup_steps     = max(100, int(0.05 * max_steps))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_steps - warmup_steps, eta_min=args.lr * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                _, _, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr  = 0
    step = 0
    t0   = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                       step=step,
                       max_steps=max_steps,
                       loss=loss.item(),
                       elapsed_time=elapsed,
                       prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                           step=step,
                           max_steps=max_steps,
                           loss=val_loss,
                           elapsed_time=elapsed)

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
