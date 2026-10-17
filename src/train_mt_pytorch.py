#!/usr/bin/env python3
"""
Modern, lightweight PyTorch implementation of an NMT Transformer (equivalent to transformer_iwslt_de_en).
Optimized for modern NVIDIA GPUs (RTX 4090, A100, H100) using:
  - PyTorch 2.x Scaled Dot-Product Attention (FlashAttention-2 / cuDNN fused kernels)
  - Native bfloat16 mixed precision
  - Pre-LN Transformer architecture for fast, stable convergence
  - Length-bucketed dynamic token batching (max-tokens)
  - Inverse sqrt learning rate schedule with linear warmup
  - Fast validation loss / perplexity early stopping
  - Fast inference & BLEU evaluation
"""

import argparse
import collections
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
        self.token2id: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }
        self.id2token: list[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.id2token)
            self.token2id[token] = idx
            self.id2token.append(token)
            return idx
        return self.token2id[token]

    def encode(self, line: str) -> list[int]:
        tokens = line.strip().split()
        return [self.token2id.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: list[int], remove_special: bool = True) -> str:
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
            f.writelines(f"{token}\n" for token in self.id2token)

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
    max_vocab_size: int | None = None,
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
# Dataset & Dynamic Length-Bucketed Batching
# -----------------------------------------------------------------------------
class ParallelDataset:
    def __init__(self, src_file: str, tgt_file: str, vocab: Vocabulary, max_seq_len: int = 256):
        self.examples: list[tuple[list[int], list[int]]] = []
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
    max_tokens: int = 8192,
    shuffle: bool = True,
    seed: int = 42,
) -> list[list[tuple[list[int], list[int]]]]:
    # Sort indices by approximate sequence length to minimize padding waste
    indices = list(range(len(dataset)))
    indices.sort(key=lambda idx: max(len(dataset.examples[idx][0]), len(dataset.examples[idx][1])))

    batches = []
    current_batch = []
    current_max_len = 0

    for idx in indices:
        s_len = len(dataset.examples[idx][0])
        t_len = len(dataset.examples[idx][1])
        item_max_len = max(s_len, t_len)

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

    # Convert indices back to items
    return [[dataset.examples[i] for i in b] for b in batches]


def collate_batch(batch_items: list[tuple[list[int], list[int]]], device: torch.device):
    src_lens = [len(x[0]) for x in batch_items]
    tgt_lens = [len(x[1]) for x in batch_items]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    B = len(batch_items)

    src_tokens = torch.full((B, max_src), PAD_ID, dtype=torch.long)
    tgt_in = torch.full((B, max_tgt + 1), PAD_ID, dtype=torch.long)
    tgt_out = torch.full((B, max_tgt + 1), PAD_ID, dtype=torch.long)

    for i, (s, t) in enumerate(batch_items):
        src_tokens[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        # Teacher forcing: tgt_in begins with <s>, tgt_out ends with </s>
        tgt_in[i, 0] = BOS_ID
        tgt_in[i, 1:len(t) + 1] = torch.tensor(t, dtype=torch.long)

        tgt_out[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        tgt_out[i, len(t)] = EOS_ID

    src_tokens = src_tokens.to(device, non_blocking=True)
    tgt_in = tgt_in.to(device, non_blocking=True)
    tgt_out = tgt_out.to(device, non_blocking=True)

    return src_tokens, tgt_in, tgt_out


# -----------------------------------------------------------------------------
# Transformer Architecture (Pre-LN with PyTorch SDPA)
# -----------------------------------------------------------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, Sq, _ = query.shape
        q = self.q_proj(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            # Incremental KV-caching during auto-regressive decoding
            k_new = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_new = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k_new], dim=2)
                v = torch.cat([kv_cache["v"], v_new], dim=2)
            else:
                k, v = k_new, v_new
            kv_cache["k"] = k
            kv_cache["v"] = v
        else:
            Sk = key.size(1)
            k = self.k_proj(key).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(value).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        combined_mask = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, Sk] where True indicates padding
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Sk]
            float_pad_mask = torch.zeros_like(pad_mask, dtype=query.dtype).masked_fill(pad_mask, float("-inf"))
            if combined_mask is None:
                combined_mask = float_pad_mask
            else:
                combined_mask = combined_mask + float_pad_mask

        dropout_p = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and (combined_mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(B, Sq, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, ffn_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_out)

        # Pre-LN Feed-Forward
        normed = self.norm2(x)
        ffn_out = self.fc2(self.dropout1(self.act(self.fc1(normed))))
        x = x + self.dropout2(ffn_out)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, ffn_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        enc_padding_mask: torch.Tensor | None = None,
        self_kv_cache: dict[str, torch.Tensor] | None = None,
        cross_kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Pre-LN Causal Self-Attention
        normed = self.norm1(x)
        self_attn_out = self.self_attn(normed, normed, normed, attn_mask=tgt_mask, kv_cache=self_kv_cache)
        x = x + self.dropout1(self_attn_out)

        # Pre-LN Cross-Attention
        normed = self.norm2(x)
        cross_attn_out = self.cross_attn(
            normed, enc_out, enc_out, key_padding_mask=enc_padding_mask, kv_cache=cross_kv_cache
        )
        x = x + self.dropout2(cross_attn_out)

        # Pre-LN Feed-Forward
        normed = self.norm3(x)
        ffn_out = self.fc2(self.dropout1(self.act(self.fc1(normed))))
        x = x + self.dropout3(ffn_out)
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        ffn_dim: int = 1024,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)

        # Shared token embeddings across encoder, decoder, and output projection
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos_emb = SinusoidalPositionalEmbedding(embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying (share decoder input and output embeddings)
        self.output_projection.weight = self.shared_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_tokens: torch.Tensor, src_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.shared_embedding(src_tokens) * self.scale
        x = self.pos_emb(x)
        x = self.emb_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=src_padding_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        enc_padding_mask: torch.Tensor | None = None,
        self_kv_caches: list[dict[str, torch.Tensor]] | None = None,
        cross_kv_caches: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        x = self.shared_embedding(tgt_tokens) * self.scale
        x = self.pos_emb(x)
        x = self.emb_dropout(x)

        for i, layer in enumerate(self.decoder_layers):
            s_cache = self_kv_caches[i] if self_kv_caches else None
            c_cache = cross_kv_caches[i] if cross_kv_caches else None
            x = layer(
                x,
                enc_out,
                tgt_mask=tgt_mask,
                enc_padding_mask=enc_padding_mask,
                self_kv_cache=s_cache,
                cross_kv_cache=c_cache,
            )

        x = self.decoder_norm(x)
        return self.output_projection(x)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        src_padding_mask = src_tokens == PAD_ID
        enc_out = self.encode(src_tokens, src_padding_mask=src_padding_mask)

        # Causal mask for teacher-forced decoder
        St = tgt_tokens.size(1)
        causal_mask = torch.triu(
            torch.full((St, St), float("-inf"), device=tgt_tokens.device, dtype=torch.float32),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(1)  # [1, 1, St, St]

        # Target padding mask
        tgt_pad_mask = (tgt_tokens == PAD_ID).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, St]
        combined_tgt_mask = causal_mask.masked_fill(tgt_pad_mask, float("-inf"))

        logits = self.decode(
            tgt_tokens,
            enc_out,
            tgt_mask=combined_tgt_mask,
            enc_padding_mask=src_padding_mask,
        )
        return logits


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
# Evaluation & Fast Generation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: list[list[tuple[list[int], list[int]]]],
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
    model: TransformerSeq2Seq,
    src_sentences: list[str],
    vocab: Vocabulary,
    device: torch.device,
    max_len: int = 120,
    batch_size: int = 128,
) -> list[str]:
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
        enc_out = model.encode(src_tokens, src_padding_mask=src_padding_mask)

        # Autoregressive decoding
        B = len(chunk)
        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            St = generated.size(1)
            causal_mask = torch.triu(
                torch.full((St, St), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(1)

            logits = model.decode(
                generated,
                enc_out,
                tgt_mask=causal_mask,
                enc_padding_mask=src_padding_mask,
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

            # Stop generating after EOS
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, PAD_ID), next_tokens)
            generated = torch.cat([generated, next_tokens], dim=1)

            finished = finished | (next_tokens.squeeze(1) == EOS_ID)
            if finished.all():
                break

        for row in range(B):
            ids = generated[row, 1:].tolist()
            translations.append(vocab.decode(ids, remove_special=True))

    return translations


def compute_simple_bleu(hypotheses: list[str], references: list[str]) -> float:
    try:
        import sacrebleu
        res = sacrebleu.corpus_bleu(hypotheses, [[r] for r in references])
        return float(res.score)
    except ImportError:
        # Fallback approximation: unigram/bigram BLEU if sacrebleu is not installed
        from collections import Counter
        clipped_1, total_1 = 0, 0
        clipped_2, total_2 = 0, 0
        hyp_len_total, ref_len_total = 0, 0

        for hyp, ref in zip(hypotheses, references):
            h_toks = hyp.split()
            r_toks = ref.split()
            hyp_len_total += len(h_toks)
            ref_len_total += len(r_toks)

            # 1-gram
            h_c1 = Counter(h_toks)
            r_c1 = Counter(r_toks)
            for w, c in h_c1.items():
                clipped_1 += min(c, r_c1.get(w, 0))
            total_1 += len(h_toks)

            # 2-gram
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
# Main Training Function
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Modern PyTorch Transformer NMT Training")
    parser.add_argument("--train-src", required=True, help="Path to tokenized training source text")
    parser.add_argument("--train-tgt", required=True, help="Path to tokenized training target text")
    parser.add_argument("--valid-src", required=True, help="Path to tokenized validation source text")
    parser.add_argument("--valid-tgt", required=True, help="Path to tokenized validation target text")
    parser.add_argument("--test-src", default=None, help="Optional path to test source text for final BLEU")
    parser.add_argument("--test-tgt", default=None, help="Optional path to test target text for final BLEU")

    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save checkpoints and vocab")
    parser.add_argument("--vocab-file", default=None, help="Pre-built vocab file; if None, built from train data")
    parser.add_argument("--max-vocab-size", type=int, default=32768, help="Max shared vocabulary size")

    # Architecture (defaults mirror transformer_iwslt_de_en)
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
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per batch (push higher on RTX 4090)")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation loss")
    parser.add_argument("--log-interval", type=int, default=500, help="Print training stats every N updates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for extra speedup")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use bfloat16 on Ampere/Ada/Hopper (RTX 4090, A100, etc.)
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Using mixed precision: {amp_dtype}")

    # 1. Build or Load Vocabulary
    vocab_path = args.vocab_file or os.path.join(args.save_dir, "vocab.txt")
    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}...")
        vocab = Vocabulary.load(vocab_path)
    else:
        print(f"Building shared vocabulary from {args.train_src} and {args.train_tgt}...")
        vocab = build_shared_vocab(args.train_src, args.train_tgt, max_vocab_size=args.max_vocab_size)
        vocab.save(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    # 2. Load Datasets
    print("Loading train dataset...")
    train_dataset = ParallelDataset(args.train_src, args.train_tgt, vocab)
    print(f"Train sentences: {len(train_dataset)}")

    print("Loading valid dataset...")
    valid_dataset = ParallelDataset(args.valid_src, args.valid_tgt, vocab)
    print(f"Valid sentences: {len(valid_dataset)}")

    valid_batches = make_batches(valid_dataset, max_tokens=args.max_tokens * 2, shuffle=False)

    # 3. Model
    model = TransformerSeq2Seq(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        ffn_dim=args.ffn_dim,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created. Total trainable parameters: {total_params / 1e6:.2f}M")

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # 4. Optimizer & Loss
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

        total_epoch_loss = 0.0
        total_epoch_tokens = 0
        running_loss = 0.0
        running_tokens = 0
        step_timer = time.time()

        for batch_idx, batch_items in enumerate(train_batches):
            global_step += 1

            # Update learning rate
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
            total_epoch_loss += loss.item() * non_pad_count
            total_epoch_tokens += non_pad_count

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

        # Epoch Validation (Fast Parallel Forward Pass)
        val_loss = evaluate_loss(model, valid_batches, criterion, device, amp_dtype)
        val_ppl = math.exp(min(20, val_loss))
        epoch_time = time.time() - epoch_start

        print(
            f"=== Epoch {epoch:2d} Complete | Time: {epoch_time:.1f}s | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} ==="
        )

        # Checkpoint selection based on validation loss
        is_best = val_loss < best_val_loss
        if is_best:
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

        # Always save last checkpoint
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

    # 5. Final Evaluation on Best Checkpoint
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

    # Quick evaluation (first 2,000 sentences if dataset is large, for fast turnaround)
    eval_subset_size = min(len(eval_srcs), 5000)
    print(f"Generating translations for {eval_subset_size} sentences...")
    hypotheses = greedy_translate(model, eval_srcs[:eval_subset_size], vocab, device)

    # Save translations
    out_path = os.path.join(args.save_dir, "eval_hypotheses.txt")
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(f"{hyp}\n" for hyp in hypotheses)

    bleu = compute_simple_bleu(hypotheses, eval_refs[:eval_subset_size])
    print(f"Final Evaluation BLEU: {bleu:.2f}")

    # Output log line compatible with 08-get_mt_bleu.py regex scraper:
    # "epoch (\d+)" ... "best_bleu (\d+\.?\d*)" ... "bleu (\d+\.?\d*)"
    print(f"epoch {best_epoch} | val_loss {best_val_loss:.4f} | bleu {bleu:.2f} | best_bleu {bleu:.2f}")


if __name__ == "__main__":
    main()
