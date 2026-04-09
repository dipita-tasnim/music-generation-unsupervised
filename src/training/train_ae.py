"""
Training script for Task 1: LSTM Autoencoder single-genre music generation.

Algorithm 1:
    1. Initialize encoder f_phi, decoder g_theta
    2. For each epoch:
        For each batch X in D:
            z = f_phi(X)
            X_hat = g_theta(z)
            L_AE = ||X - X_hat||^2
            Update (phi, theta) <- (phi, theta) - eta * grad(L_AE)
    3. Generate new music by sampling latent codes z
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    AE_LATENT_DIM, AE_HIDDEN_DIM, AE_NUM_LAYERS, AE_DROPOUT,
    AE_LEARNING_RATE, AE_BATCH_SIZE, AE_EPOCHS,
    NUM_PITCHES, SEQUENCE_LENGTH, DEVICE, RANDOM_SEED,
    PROCESSED_DIR, CHECKPOINT_DIR, PLOTS_DIR, GENERATED_MIDI_DIR,
    TRAIN_RATIO,
)
from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.piano_roll import build_piano_roll_dataset
from src.preprocessing.midi_parser import load_parsed_data


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch["piano_roll"].to(device)  # (B, T, 128)
        optimizer.zero_grad()

        x_hat, z = model(x)
        loss = model.reconstruction_loss(x, x_hat)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["piano_roll"].to(device)
            x_hat, z = model(x)
            loss = model.reconstruction_loss(x, x_hat)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Load data ---
    parsed_path = os.path.join(PROCESSED_DIR, "parsed_midi.json")
    if os.path.exists(parsed_path):
        parsed_data = load_parsed_data(parsed_path)
        dataset = build_piano_roll_dataset(parsed_data, seg_len=args.seq_len)
    else:
        print("[INFO] No parsed data found. Creating synthetic dataset for demonstration.")
        from src.preprocessing.piano_roll import PianoRollDataset
        segments = [np.random.rand(args.seq_len, NUM_PITCHES).astype(np.float32) * 0.3 for _ in range(500)]
        dataset = PianoRollDataset(segments)

    # --- Cap dataset size for CPU training ---
    if args.max_samples and len(dataset) > args.max_samples:
        dataset, _ = random_split(dataset, [args.max_samples, len(dataset) - args.max_samples],
                                  generator=torch.Generator().manual_seed(RANDOM_SEED))
        print(f"[INFO] Dataset capped to {args.max_samples} samples for faster training.")

    # --- Train / Val split ---
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = LSTMAutoencoder(
        input_dim=NUM_PITCHES,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_set)} samples, validating on {len(val_set)} samples")
    print(f"Device: {DEVICE}")

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "autoencoder_best.pt"))

    # Save final model & history
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "autoencoder_final.pt"))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(os.path.join(PLOTS_DIR, "ae_training_history.json"), "w") as f:
        json.dump(history, f)

    # --- Plot loss curve ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss (MSE)")
        plt.title("Task 1: LSTM Autoencoder Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "ae_loss_curve.png"), dpi=150)
        plt.close()
        print(f"Loss curve saved to {PLOTS_DIR}/ae_loss_curve.png")
    except ImportError:
        print("[WARN] matplotlib not available, skipping loss plot.")

    # --- Generate samples ---
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "autoencoder_best.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.eval()

    os.makedirs(GENERATED_MIDI_DIR, exist_ok=True)
    from src.generation.midi_export import piano_roll_to_midi

    # Encode real data, perturb latent codes, decode for structured output
    sample_loader = DataLoader(train_set, batch_size=5, shuffle=True)
    real_batch = next(iter(sample_loader))["piano_roll"].to(DEVICE)
    with torch.no_grad():
        z = model.encode(real_batch)
        # Add small noise for variation
        z_perturbed = z + torch.randn_like(z) * 0.3
        generated = model.decode(z_perturbed)  # (5, seq_len, 128)

    for i in range(generated.size(0)):
        roll = generated[i].cpu().numpy()
        midi_path = os.path.join(GENERATED_MIDI_DIR, f"ae_generated_{i+1}.mid")
        piano_roll_to_midi(roll, midi_path, threshold=0.3)
    print(f"Generated 5 MIDI samples in {GENERATED_MIDI_DIR}")

    print("Task 1 training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder (Task 1)")
    parser.add_argument("--epochs", type=int, default=AE_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=AE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=AE_LEARNING_RATE)
    parser.add_argument("--hidden_dim", type=int, default=AE_HIDDEN_DIM)
    parser.add_argument("--latent_dim", type=int, default=AE_LATENT_DIM)
    parser.add_argument("--num_layers", type=int, default=AE_NUM_LAYERS)
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--dropout", type=float, default=AE_DROPOUT)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for CPU training")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
