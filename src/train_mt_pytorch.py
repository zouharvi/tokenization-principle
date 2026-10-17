#!/usr/bin/env python3
"""
Modern, lightweight PyTorch NMT training script using built-in torch.nn.Transformer.
Equivalent in size and parameters to transformer_iwslt_de_en (Pre-LN).
Optimized for modern GPUs (RTX 4090, A100, H100) using:
  - Built-in torch.nn.Transformer with PyTorch 2.x SDPA (FlashAttention-2 / cuDNN)
  - Native bfloat16 mixed precision
  - Dynamic token batching (max-tokens)
  - Inverse sqrt learning rate schedule with linear warmup
  - Fast validation loss / perplexity early stopping
"""

import argparse
import collections
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------
class Vocabulary:
    def __init__(self):
        self.token2id: Dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }
        self.id2token: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.id2token)
            self.token2id[token] = idx
            self.id2token.append(token)
            return idx
        return self.token2id[token]

    def encode(self, line: str) -> List[int]:
        tokens = line.strip().split()
        return [self.token2id.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        tokens = []
        for i in ids:
            if remove_special and i in (PAD_ID, BOS_ID, EOS_ID):
                if i == EOS_ID:
                    break
                continue
            tokens.append(self.id2token[i] if i < len(self.id2token) else UNK_TOKEN)
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.id2token)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for token in self.id2token:
                f.write(f"{token}\n")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        vocab.token2id.clear()
        vocab.id2token.clear()
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\r\n")
                vocab.token2id[token] = idx
                vocab.id2token.append(token)
        return vocab


def build_shared_vocab(
    src_file: str,
    tgt_file: str,
    max_vocab_size: Optional[int] = None,
    min_freq: int = 1,
) -> Vocabulary:
    counts = collections.Counter()
    for filepath in (src_file, tgt_file):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                counts.update(line.strip().split())

    vocab = Vocabulary()
    sorted_tokens = [tok for tok, count in counts.most_common() if count >= min_freq]
    if max_vocab_size is not None:
        remaining_slots = max(0, max_vocab_size - len(vocab))
        sorted_tokens = sorted_tokens[:remaining_slots]

    for tok in sorted_tokens:
        vocab.add_token(tok)

    return vocab


# -----------------------------------------------------------------------------
# Dataset & Dynamic Token Batching
# -----------------------------------------------------------------------------
class ParallelDataset:
    def __init__(self, src_file: str, tgt_file: str, vocab: Vocabulary, max_seq_len: int = 256):
        self.examples: List[Tuple[List[int], List[int]]] = []
        with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
            for line_src, line_tgt in zip(f_src, f_tgt):
                s_ids = vocab.encode(line_src)
                t_ids = vocab.encode(line_tgt)
                if 0 < len(s_ids) <= max_seq_len and 0 < len(t_ids) <= max_seq_len:
                    self.examples.append((s_ids, t_ids))

    def __len__(self) -> int:
        return len(self.examples)


def make_batches(
    dataset: ParallelDataset,
    max_tokens: int = 12288,
    shuffle: bool = True,
    seed: int = 42,
) -> List[List[Tuple[List[int], List[int]]]]:
    # Sort indices by maximum sequence length to minimize padding
    indices = list(range(len(dataset)))
    indices.sort(key=lambda idx: max(len(dataset.examples[idx][0]), len(dataset.examples[idx][1])))

    batches = []
    current_batch = []
    current_max_len = 0

    for idx in indices:
        item_max_len = max(len(dataset.examples[idx][0]), len(dataset.examples[idx][1]))
        projected_max_len = max(current_max_len, item_max_len)
        projected_tokens = (len(current_batch) + 1) * projected_max_len

        if current_batch and projected_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = [idx]
            current_max_len = item_max_len
        else:
            current_batch.append(idx)
            current_max_len = projected_max_len

    if current_batch:
        batches.append(current_batch)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(batches)

    return [[dataset.examples[i] for i in b] for b in batches]


def collate_batch(batch_items: List[Tuple[List[int], List[int]]], device: torch.device):
    max_src = max(len(x[0]) for x in batch_items)
    max_tgt = max(len(x[1]) for x in batch_items)
    B = len(batch_items)

    src_tokens = torch.full((B, max_src), PAD_ID, dtype=torch.long)
    tgt_in = torch.full((B, max_tgt + 1), PAD_ID, dtype=torch.long)
    tgt_out = torch.full((B, max_tgt + 1), PAD_ID, dtype=torch.long)

    for i, (s, t) in enumerate(batch_items):
        src_tokens[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_in[i, 0] = BOS_ID
        tgt_in[i, 1:len(t) + 1] = torch.tensor(t, dtype=torch.long)
        tgt_out[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        tgt_out[i, len(t)] = EOS_ID

    return (
        src_tokens.to(device, non_blocking=True),
        tgt_in.to(device, non_blocking=True),
        tgt_out.to(device, non_blocking=True),
    )


# -----------------------------------------------------------------------------
# Built-in PyTorch Transformer Model Wrapper
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class Seq2SeqTransformer(nn.Module):
    """
    Wraps built-in torch.nn.Transformer with token embeddings, positional encodings,
    and a weight-tied language model head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Built-in torch.nn.Transformer (Pre-LN: norm_first=True)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (share decoder input and output embeddings)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_emb = self.emb_dropout(self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model)))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_emb = self.emb_dropout(self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model)))
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.lm_head(out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src == PAD_ID
        tgt_key_padding_mask = tgt == PAD_ID

        tgt_seq_len = tgt.size(1)
        # Built-in subsequent causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=tgt.device)

        src_emb = self.emb_dropout(self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.emb_dropout(self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model)))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.lm_head(out)


# -----------------------------------------------------------------------------
# Learning Rate Scheduler (Inverse Sqrt with Linear Warmup)
# -----------------------------------------------------------------------------
def get_inverse_sqrt_lr(step: int, warmup_updates: int, base_lr: float) -> float:
    if step < 1:
        return 0.0
    if step <= warmup_updates:
        return base_lr * (step / float(max(1, warmup_updates)))
    return base_lr * (warmup_updates / float(step)) ** 0.5


# -----------------------------------------------------------------------------
# Evaluation & Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: List[List[Tuple[List[int], List[int]]]],
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_items in batches:
        src_tokens, tgt_in, tgt_out = collate_batch(batch_items, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(src_tokens, tgt_in)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))

        non_pad_count = (tgt_out != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad_count
        total_tokens += non_pad_count

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def greedy_translate(
    model: Seq2SeqTransformer,
    src_sentences: List[str],
    vocab: Vocabulary,
    device: torch.device,
    max_len: int = 120,
    batch_size: int = 128,
) -> List[str]:
    model.eval()
    translations = []

    for i in range(0, len(src_sentences), batch_size):
        chunk = src_sentences[i : i + batch_size]
        encoded = [vocab.encode(s) for s in chunk]
        max_src = max(len(s) for s in encoded)
        src_tokens = torch.full((len(chunk), max_src), PAD_ID, dtype=torch.long, device=device)
        for row, s in enumerate(encoded):
            src_tokens[row, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        src_padding_mask = src_tokens == PAD_ID
        memory = model.encode(src_tokens, src_key_padding_mask=src_padding_mask)

        B = len(chunk)
        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=device)
            logits = model.decode(
                generated,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, PAD_ID), next_tokens)
            generated = torch.cat([generated, next_tokens], dim=1)

            finished = finished | (next_tokens.squeeze(1) == EOS_ID)
            if finished.all():
                break

        for row in range(B):
            ids = generated[row, 1:].tolist()
            translations.append(vocab.decode(ids, remove_special=True))

    return translations


def compute_simple_bleu(hypotheses: List[str], references: List[str]) -> float:
    try:
        import sacrebleu
        res = sacrebleu.corpus_bleu(hypotheses, [[r] for r in references])
        return float(res.score)
    except ImportError:
        from collections import Counter
        clipped_1, total_1 = 0, 0
        clipped_2, total_2 = 0, 0
        hyp_len_total, ref_len_total = 0, 0

        for hyp, ref in zip(hypotheses, references):
            h_toks = hyp.split()
            r_toks = ref.split()
            hyp_len_total += len(h_toks)
            ref_len_total += len(r_toks)

            h_c1 = Counter(h_toks)
            r_c1 = Counter(r_toks)
            for w, c in h_c1.items():
                clipped_1 += min(c, r_c1.get(w, 0))
            total_1 += len(h_toks)

            h_2 = [f"{h_toks[i]}_{h_toks[i+1]}" for i in range(len(h_toks) - 1)]
            r_2 = [f"{r_toks[i]}_{r_toks[i+1]}" for i in range(len(r_toks) - 1)]
            h_c2 = Counter(h_2)
            r_c2 = Counter(r_2)
            for w, c in h_c2.items():
                clipped_2 += min(c, r_c2.get(w, 0))
            total_2 += len(h_2)

        if total_1 == 0 or clipped_1 == 0 or total_2 == 0 or clipped_2 == 0:
            return 0.0

        p1 = clipped_1 / total_1
        p2 = clipped_2 / total_2
        bp = math.exp(min(0.0, 1.0 - (ref_len_total / max(1, hyp_len_total))))
        return float(bp * math.sqrt(p1 * p2) * 100.0)


# -----------------------------------------------------------------------------
# Training Entry Point
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch NMT Training with built-in torch.nn.Transformer")
    parser.add_argument("--train-src", required=True, help="Tokenized train source file")
    parser.add_argument("--train-tgt", required=True, help="Tokenized train target file")
    parser.add_argument("--valid-src", required=True, help="Tokenized valid source file")
    parser.add_argument("--valid-tgt", required=True, help="Tokenized valid target file")
    parser.add_argument("--test-src", default=None, help="Optional test source file")
    parser.add_argument("--test-tgt", default=None, help="Optional test target file")

    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--vocab-file", default=None)
    parser.add_argument("--max-vocab-size", type=int, default=32768)

    # Architecture (mirrors transformer_iwslt_de_en)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Optimization
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-updates", type=int, default=4000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    # Training loop
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Using mixed precision: {amp_dtype}")

    # Vocabulary
    vocab_path = args.vocab_file or os.path.join(args.save_dir, "vocab.txt")
    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}...")
        vocab = Vocabulary.load(vocab_path)
    else:
        print(f"Building shared vocabulary from {args.train_src} and {args.train_tgt}...")
        vocab = build_shared_vocab(args.train_src, args.train_tgt, max_vocab_size=args.max_vocab_size)
        vocab.save(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    # Datasets
    print("Loading train dataset...")
    train_dataset = ParallelDataset(args.train_src, args.train_tgt, vocab)
    print(f"Train sentences: {len(train_dataset)}")

    print("Loading valid dataset...")
    valid_dataset = ParallelDataset(args.valid_src, args.valid_tgt, vocab)
    print(f"Valid sentences: {len(valid_dataset)}")

    valid_batches = make_batches(valid_dataset, max_tokens=args.max_tokens * 2, shuffle=False)

    # Built-in PyTorch Transformer Model
    model = Seq2SeqTransformer(
        vocab_size=len(vocab),
        d_model=args.embed_dim,
        nhead=args.num_heads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created using torch.nn.Transformer. Trainable parameters: {total_params / 1e6:.2f}M")

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Optimizer & Loss
    fused_adam = torch.cuda.is_available() and hasattr(torch.optim.AdamW, "_fused")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=fused_adam,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=PAD_ID)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    best_epoch = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        model.train()

        train_batches = make_batches(
            train_dataset,
            max_tokens=args.max_tokens,
            shuffle=True,
            seed=args.seed + epoch,
        )

        running_loss = 0.0
        running_tokens = 0
        step_timer = time.time()

        for batch_idx, batch_items in enumerate(train_batches):
            global_step += 1

            lr = get_inverse_sqrt_lr(global_step, args.warmup_updates, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            src_tokens, tgt_in, tgt_out = collate_batch(batch_items, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(src_tokens, tgt_in)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            non_pad_count = (tgt_out != PAD_ID).sum().item()
            running_loss += loss.item() * non_pad_count
            running_tokens += non_pad_count

            if global_step % args.log_interval == 0:
                elapsed = time.time() - step_timer
                tok_per_sec = running_tokens / max(1e-5, elapsed)
                avg_step_loss = running_loss / max(1, running_tokens)
                print(
                    f"Epoch {epoch:2d} | Step {global_step:6d} | lr {lr:.2e} | "
                    f"loss {avg_step_loss:.4f} | ppl {math.exp(min(20, avg_step_loss)):.2f} | "
                    f"{tok_per_sec:,.0f} tok/s"
                )
                running_loss = 0.0
                running_tokens = 0
                step_timer = time.time()

        # Validation Loss
        val_loss = evaluate_loss(model, valid_batches, criterion, device, amp_dtype)
        val_ppl = math.exp(min(20, val_loss))
        epoch_time = time.time() - epoch_start

        print(
            f"=== Epoch {epoch:2d} Complete | Time: {epoch_time:.1f}s | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} ==="
        )

        # Checkpoint selection
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                os.path.join(args.save_dir, "checkpoint_best.pt"),
            )
            print(f"[*] Saved new best checkpoint (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"[!] No improvement. Patience: {patience_counter}/{args.patience}")

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(args.save_dir, "checkpoint_last.pt"),
        )

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time / 60:.2f} minutes.")

    # Final Evaluation on Best Checkpoint
    best_ckpt_path = os.path.join(args.save_dir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt_path):
        print(f"Loading best checkpoint from epoch {best_epoch} for final evaluation...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    eval_src_file = args.test_src or args.valid_src
    eval_tgt_file = args.test_tgt or args.valid_tgt

    with open(eval_src_file, "r", encoding="utf-8") as f_src, open(eval_tgt_file, "r", encoding="utf-8") as f_tgt:
        eval_srcs = [l.strip() for l in f_src]
        eval_refs = [l.strip() for l in f_tgt]

    eval_subset_size = min(len(eval_srcs), 5000)
    print(f"Generating translations for {eval_subset_size} sentences...")
    hypotheses = greedy_translate(model, eval_srcs[:eval_subset_size], vocab, device)

    out_path = os.path.join(args.save_dir, "eval_hypotheses.txt")
    with open(out_path, "w", encoding="utf-8") as f_out:
        for hyp in hypotheses:
            f_out.write(f"{hyp}\n")

    bleu = compute_simple_bleu(hypotheses, eval_refs[:eval_subset_size])
    print(f"Final Evaluation BLEU: {bleu:.2f}")

    # Output log line compatible with 08-get_mt_bleu.py regex scraper:
    print(f"epoch {best_epoch} | val_loss {best_val_loss:.4f} | bleu {bleu:.2f} | best_bleu {bleu:.2f}")


if __name__ == "__main__":
    main()
