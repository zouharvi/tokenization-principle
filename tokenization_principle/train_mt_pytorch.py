#!/usr/bin/env python3
"""
Minimal PyTorch NMT training script using built-in torch.nn.Transformer and sacrebleu chrF.
"""

import argparse
import collections
import math
import os
import random
import time

import sacrebleu
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class Vocabulary:
    def __init__(self):
        self.token2id = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<s>": BOS_ID, "</s>": EOS_ID}
        self.id2token = ["<pad>", "<unk>", "<s>", "</s>"]

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)
        return self.token2id[token]

    def encode(self, line: str):
        return [self.token2id.get(t, UNK_ID) for t in line.strip().split()]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == EOS_ID:
                break
            if i not in (PAD_ID, BOS_ID):
                tokens.append(self.id2token[i] if i < len(self.id2token) else "<unk>")
        return " ".join(tokens)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.id2token))

    @classmethod
    def load(cls, path: str):
        v = cls()
        with open(path, "r", encoding="utf-8") as f:
            v.id2token = [line.rstrip("\r\n") for line in f]
        v.token2id = {t: i for i, t in enumerate(v.id2token)}
        return v

    def __len__(self):
        return len(self.id2token)


def build_vocab(src_path: str, tgt_path: str, max_size: int = 32768):
    counts = collections.Counter()
    for p in (src_path, tgt_path):
        with open(p, "r", encoding="utf-8") as f:
            for l in f:
                counts.update(l.strip().split())
    v = Vocabulary()
    for tok, _ in counts.most_common(max(0, max_size - len(v))):
        v.add_token(tok)
    return v


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        max_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

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
        self.lm_head.weight = self.embedding.weight

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        pos = torch.arange(src.size(1), device=src.device)
        x = self.emb_dropout(self.embedding(src) * math.sqrt(self.d_model) + self.pos_emb(pos))
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ):
        pos = torch.arange(tgt.size(1), device=tgt.device)
        x = self.emb_dropout(self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_emb(pos))
        out = self.transformer.decoder(
            x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask
        )
        return self.lm_head(out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = src == PAD_ID
        tgt_pad_mask = tgt == PAD_ID
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

        pos_src = torch.arange(src.size(1), device=src.device)
        pos_tgt = torch.arange(tgt.size(1), device=tgt.device)
        src_emb = self.emb_dropout(self.embedding(src) * math.sqrt(self.d_model) + self.pos_emb(pos_src))
        tgt_emb = self.emb_dropout(self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_emb(pos_tgt))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )
        return self.lm_head(out)


@torch.no_grad()
def greedy_translate(model, sentences, vocab, device, max_len=120, batch_size=128):
    model.eval()
    translations = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        encoded = [vocab.encode(s) for s in chunk]
        max_src = max(len(s) for s in encoded)
        src = torch.full((len(chunk), max_src), PAD_ID, dtype=torch.long, device=device)
        for row, s in enumerate(encoded):
            src[row, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        src_mask = src == PAD_ID
        memory = model.encode(src, src_key_padding_mask=src_mask)

        gen = torch.full((len(chunk), 1), BOS_ID, dtype=torch.long, device=device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(gen.size(1), device=device)
            logits = model.decode(gen, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_tok = torch.where(done.unsqueeze(1), torch.full_like(next_tok, PAD_ID), next_tok)
            gen = torch.cat([gen, next_tok], dim=1)
            done = done | (next_tok.squeeze(1) == EOS_ID)
            if done.all():
                break

        for row in range(len(chunk)):
            translations.append(vocab.decode(gen[row, 1:].tolist()))
    return translations


def load_pairs(src_file: str, tgt_file: str, vocab: Vocabulary, max_len: int = 256):
    pairs = []
    with open(src_file, "r", encoding="utf-8") as fs, open(tgt_file, "r", encoding="utf-8") as ft:
        for ls, lt in zip(fs, ft):
            s_ids, t_ids = vocab.encode(ls), vocab.encode(lt)
            if 0 < len(s_ids) <= max_len and 0 < len(t_ids) <= max_len:
                pairs.append((s_ids, t_ids))
    return pairs


def make_batches(pairs, max_tokens: int = 12288, shuffle: bool = True):
    sorted_pairs = sorted(pairs, key=lambda x: max(len(x[0]), len(x[1])))
    batches, cur_batch, cur_len = [], [], 0
    for item in sorted_pairs:
        l = max(len(item[0]), len(item[1]))
        if cur_batch and (len(cur_batch) + 1) * max(cur_len, l) > max_tokens:
            batches.append(cur_batch)
            cur_batch, cur_len = [item], l
        else:
            cur_batch.append(item)
            cur_len = max(cur_len, l)
    if cur_batch:
        batches.append(cur_batch)
    if shuffle:
        random.shuffle(batches)
    return batches


def collate(batch, device):
    max_s = max(len(x[0]) for x in batch)
    max_t = max(len(x[1]) for x in batch)
    B = len(batch)
    src = torch.full((B, max_s), PAD_ID, dtype=torch.long, device=device)
    tgt_in = torch.full((B, max_t + 1), PAD_ID, dtype=torch.long, device=device)
    tgt_out = torch.full((B, max_t + 1), PAD_ID, dtype=torch.long, device=device)
    for i, (s, t) in enumerate(batch):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_in[i, 0] = BOS_ID
        tgt_in[i, 1:len(t) + 1] = torch.tensor(t, dtype=torch.long)
        tgt_out[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        tgt_out[i, len(t)] = EOS_ID
    return src, tgt_in, tgt_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-src", required=True)
    parser.add_argument("--train-tgt", required=True)
    parser.add_argument("--valid-src", required=True)
    parser.add_argument("--valid-tgt", required=True)
    parser.add_argument("--test-src", default=None)
    parser.add_argument("--test-tgt", default=None)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--vocab-file", default=None)
    parser.add_argument("--max-vocab-size", type=int, default=32768)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-updates", type=int, default=4000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Vocab & Data
    vocab_path = args.vocab_file or os.path.join(args.save_dir, "vocab.txt")
    vocab = Vocabulary.load(vocab_path) if os.path.exists(vocab_path) else build_vocab(args.train_src, args.train_tgt, args.max_vocab_size)
    vocab.save(vocab_path)
    train_pairs = load_pairs(args.train_src, args.train_tgt, vocab)

    with open(args.valid_src, "r", encoding="utf-8") as fs, open(args.valid_tgt, "r", encoding="utf-8") as ft:
        valid_srcs = [l.strip() for l in fs]
        valid_refs = [l.strip() for l in ft]

    # Model
    model = Seq2SeqTransformer(
        vocab_size=len(vocab),
        d_model=args.embed_dim,
        nhead=args.num_heads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    # Optimizer & Built-in LambdaLR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8,
        weight_decay=args.weight_decay, fused=torch.cuda.is_available()
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / args.warmup_updates, (args.warmup_updates / (step + 1)) ** 0.5) if step > 0 else 0.0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=PAD_ID)

    best_chrf = -1.0
    best_epoch = 0
    patience_cnt = 0
    global_step = 0

    print(f"Starting training on {device} ({len(train_pairs)} pairs)...")
    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_batches = make_batches(train_pairs, max_tokens=args.max_tokens, shuffle=True)
        running_loss, running_toks, step_t = 0.0, 0, time.time()

        for batch_items in train_batches:
            global_step += 1
            src, tgt_in, tgt_out = collate(batch_items, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(src, tgt_in)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            scheduler.step()

            non_pad = (tgt_out != PAD_ID).sum().item()
            running_loss += loss.item() * non_pad
            running_toks += non_pad

            if global_step % args.log_interval == 0:
                tok_s = running_toks / max(1e-5, time.time() - step_t)
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:2d} | Step {global_step:6d} | lr {cur_lr:.2e} | loss {running_loss/max(1, running_toks):.4f} | {tok_s:,.0f} tok/s")
                running_loss, running_toks, step_t = 0.0, 0, time.time()

        # Validation: Evaluate chrF via sacrebleu instead of loss
        eval_n = min(len(valid_srcs), 2000)
        hypotheses = greedy_translate(model, valid_srcs[:eval_n], vocab, device)
        val_chrf = sacrebleu.corpus_chrf(hypotheses, [valid_refs[:eval_n]]).score
        print(f"=== Epoch {epoch:2d} | Time: {time.time()-epoch_start:.1f}s | Val chrF: {val_chrf:.2f} ===")

        if val_chrf > best_chrf:
            best_chrf = val_chrf
            best_epoch = epoch
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_chrf": val_chrf,
                "args": vars(args),
            }, os.path.join(args.save_dir, "checkpoint_best.pt"))
            print(f"[*] New best checkpoint (chrF: {val_chrf:.2f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Final test evaluation
    best_ckpt = os.path.join(args.save_dir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])

    test_src = args.test_src or args.valid_src
    test_tgt = args.test_tgt or args.valid_tgt
    with open(test_src, "r", encoding="utf-8") as fs, open(test_tgt, "r", encoding="utf-8") as ft:
        ts_src = [l.strip() for l in fs]
        ts_ref = [l.strip() for l in ft]

    test_n = min(len(ts_src), 5000)
    test_hyps = greedy_translate(model, ts_src[:test_n], vocab, device)
    test_chrf = sacrebleu.corpus_chrf(test_hyps, [ts_ref[:test_n]]).score

    with open(os.path.join(args.save_dir, "eval_hypotheses.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_hyps) + "\n")

    print(f"epoch {best_epoch} | chrf {test_chrf:.2f} | best_chrf {best_chrf:.2f}")


if __name__ == "__main__":
    main()
