"""
Training script for Task 3: Transformer-Based Music Generator.

Algorithm 3:
    1. Initialize Transformer model p_theta
    2. For each epoch:
        For each sequence X:
            For time step t = 1 to T:
                Predict p_theta(x_t | x_{<t})
            L_TR = -sum_t log p_theta(x_t | x_{<t})
            Update theta <- theta - eta * grad(L_TR)
    3. Generate long compositions by iterative sampling: x_t ~ p_theta(x_t | x_{<t})
"""
import os
import sys
import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    VOCAB_SIZE, TF_D_MODEL, TF_N_HEADS, TF_NUM_LAYERS, TF_D_FF,
    TF_DROPOUT, TF_LEARNING_RATE, TF_BATCH_SIZE, TF_EPOCHS,
    TF_WARMUP_STEPS, TF_MAX_SEQ_LEN, NUM_GENRES, PAD_TOKEN,
    SEQUENCE_LENGTH, DEVICE, RANDOM_SEED,
    PROCESSED_DIR, CHECKPOINT_DIR, PLOTS_DIR, GENERATED_MIDI_DIR,
    TRAIN_RATIO, BOS_TOKEN, GENRES,
)
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import build_dataset_from_parsed, MusicTokenDataset
from src.preprocessing.midi_parser import load_parsed_data


class TransformerLRScheduler:
    """Noam-style learning rate scheduler with warm-up."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def train_one_epoch(model, dataloader, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        tokens = batch["tokens"].to(device)  # (B, T)
        # Input: all tokens except last; Target: all tokens except first
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]

        genre = batch.get("genre")
        if genre is not None:
            genre = torch.clamp(genre.to(device), min=0)

        optimizer.zero_grad()
        logits = model(input_tokens, genre)  # (B, T-1, V)
        loss = model.compute_loss(logits, target_tokens)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        # Count non-pad tokens for perplexity
        non_pad = (target_tokens != PAD_TOKEN).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # clamp to avoid overflow
    return avg_loss, perplexity


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            genre = batch.get("genre")
            if genre is not None:
                genre = torch.clamp(genre.to(device), min=0)

            logits = model(input_tokens, genre)
            loss = model.compute_loss(logits, target_tokens)

            non_pad = (target_tokens != PAD_TOKEN).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity


def train(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Load data ---
    parsed_path = os.path.join(PROCESSED_DIR, "parsed_midi.json")
    if os.path.exists(parsed_path):
        parsed_data = load_parsed_data(parsed_path)
        dataset = build_dataset_from_parsed(parsed_data, seq_len=args.seq_len)
    else:
        print("[INFO] No parsed data found. Creating synthetic token dataset for demonstration.")
        sequences = []
        genre_ids = []
        for i in range(600):
            seq = [BOS_TOKEN] + [np.random.randint(3, VOCAB_SIZE) for _ in range(args.seq_len - 2)] + [2]
            sequences.append(seq)
            genre_ids.append(i % NUM_GENRES)
        dataset = MusicTokenDataset(sequences, genre_ids)

    # --- Cap dataset size for CPU training ---
    if args.max_samples and len(dataset) > args.max_samples:
        dataset, _ = random_split(dataset, [args.max_samples, len(dataset) - args.max_samples],
                                  generator=torch.Generator().manual_seed(RANDOM_SEED))
        print(f"[INFO] Dataset capped to {args.max_samples} samples for faster training.")

    # --- Split ---
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = MusicTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        num_genres=NUM_GENRES,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    lr_scheduler = TransformerLRScheduler(optimizer, args.d_model, args.warmup_steps)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_set)} samples, validating on {len(val_set)} samples")
    print(f"Device: {DEVICE}")

    # --- Training ---
    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}
    best_val_loss = float("inf")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = train_one_epoch(model, train_loader, optimizer, lr_scheduler, DEVICE)
        val_loss, val_ppl = validate(model, val_loader, DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | Train Loss: {train_loss:.4f} | PPL: {train_ppl:.2f} | "
                  f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "transformer_best.pt"))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "transformer_final.pt"))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(os.path.join(PLOTS_DIR, "transformer_training_history.json"), "w") as f:
        json.dump(history, f)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].set_title("Autoregressive Loss (L_TR)")
        axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(history["train_ppl"], label="Train Perplexity")
        axes[1].plot(history["val_ppl"], label="Val Perplexity")
        axes[1].set_title("Perplexity = exp(L_TR / T)")
        axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "transformer_training.png"), dpi=150)
        plt.close()
        print(f"Training plots saved to {PLOTS_DIR}/transformer_training.png")
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")

    # --- Generate 10 long compositions ---
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "transformer_best.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.eval()

    os.makedirs(GENERATED_MIDI_DIR, exist_ok=True)
    from src.generation.midi_export import tokens_to_midi

    for i in range(10):
        genre_idx = i % NUM_GENRES
        genre_tensor = torch.tensor([genre_idx], dtype=torch.long).to(DEVICE)
        prompt = torch.tensor([[BOS_TOKEN]], dtype=torch.long).to(DEVICE)

        generated_tokens = model.generate(prompt, max_len=256, genre=genre_tensor,
                                            temperature=0.9, top_k=30)
        token_list = generated_tokens[0].cpu().tolist()
        midi_path = os.path.join(GENERATED_MIDI_DIR, f"transformer_generated_{GENRES[genre_idx]}_{i+1}.mid")
        tokens_to_midi(token_list, midi_path)

    print(f"Generated 10 long-sequence compositions in {GENERATED_MIDI_DIR}")
    print("Task 3 training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Music Transformer (Task 3)")
    parser.add_argument("--epochs", type=int, default=TF_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TF_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TF_LEARNING_RATE)
    parser.add_argument("--d_model", type=int, default=TF_D_MODEL)
    parser.add_argument("--n_heads", type=int, default=TF_N_HEADS)
    parser.add_argument("--num_layers", type=int, default=TF_NUM_LAYERS)
    parser.add_argument("--d_ff", type=int, default=TF_D_FF)
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--dropout", type=float, default=TF_DROPOUT)
    parser.add_argument("--warmup_steps", type=int, default=TF_WARMUP_STEPS)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for CPU training")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
