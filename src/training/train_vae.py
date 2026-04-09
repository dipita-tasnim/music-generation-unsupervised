"""
Training script for Task 2: VAE Multi-Genre Music Generator.

Algorithm 2:
    1. Initialize encoder q_phi(z|X), decoder p_theta(X|z)
    2. For each epoch:
        For each batch X:
            (μ, σ) = Encoder(X)
            z = μ + σ ⊙ ε,  ε ~ N(0, I)
            X_hat = Decoder(z)
            L_recon = ||X - X_hat||^2
            L_KL = D_KL(q_phi(z|X) || p(z))
            L_VAE = L_recon + β * L_KL
            Update (phi, theta)
    3. Generate diverse multi-genre music by sampling z ~ N(0, I)
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
    VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_NUM_LAYERS, VAE_DROPOUT,
    VAE_LEARNING_RATE, VAE_BATCH_SIZE, VAE_EPOCHS, VAE_BETA,
    VAE_BETA_ANNEAL_EPOCHS, NUM_PITCHES, SEQUENCE_LENGTH,
    NUM_GENRES, DEVICE, RANDOM_SEED,
    PROCESSED_DIR, CHECKPOINT_DIR, PLOTS_DIR, GENERATED_MIDI_DIR,
    TRAIN_RATIO, GENRES,
)
from src.models.vae import MusicVAE
from src.preprocessing.piano_roll import build_piano_roll_dataset
from src.preprocessing.midi_parser import load_parsed_data


def get_beta_schedule(epoch: int, max_beta: float, anneal_epochs: int) -> float:
    """Linear β annealing warm-up for KL divergence."""
    return min(max_beta, max_beta * epoch / max(anneal_epochs, 1))


def train_one_epoch(model, dataloader, optimizer, device, beta):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch["piano_roll"].to(device)
        genre = batch.get("genre")
        if genre is not None:
            genre = genre.to(device)
            # Filter out unknown genres (-1)
            valid = genre >= 0
            if valid.sum() == 0:
                genre = None
            else:
                genre = torch.clamp(genre, min=0)

        optimizer.zero_grad()
        x_hat, mu, logvar, z = model(x, genre)
        losses = model.loss_function(x, x_hat, mu, logvar, beta=beta)

        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["loss"].item()
        total_recon += losses["recon_loss"].item()
        total_kl += losses["kl_loss"].item()
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_recon / n, total_kl / n


def validate(model, dataloader, device, beta):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["piano_roll"].to(device)
            genre = batch.get("genre")
            if genre is not None:
                genre = torch.clamp(genre.to(device), min=0)

            x_hat, mu, logvar, z = model(x, genre)
            losses = model.loss_function(x, x_hat, mu, logvar, beta=beta)

            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl += losses["kl_loss"].item()
            num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_recon / n, total_kl / n


def train(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Load data ---
    parsed_path = os.path.join(PROCESSED_DIR, "parsed_midi.json")
    if os.path.exists(parsed_path):
        parsed_data = load_parsed_data(parsed_path)
        dataset = build_piano_roll_dataset(parsed_data, seg_len=args.seq_len)
    else:
        print("[INFO] No parsed data found. Creating synthetic multi-genre dataset for demonstration.")
        from src.preprocessing.piano_roll import PianoRollDataset
        segments = [np.random.rand(args.seq_len, NUM_PITCHES).astype(np.float32) * 0.3 for _ in range(800)]
        genre_ids = [i % NUM_GENRES for i in range(800)]
        dataset = PianoRollDataset(segments, genre_ids)

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
    model = MusicVAE(
        input_dim=NUM_PITCHES,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        dropout=args.dropout,
        num_genres=NUM_GENRES,
        beta=args.beta,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_set)} samples, validating on {len(val_set)} samples")
    print(f"Device: {DEVICE}")

    # --- Training ---
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": [], "beta": []}
    best_val_loss = float("inf")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        beta = get_beta_schedule(epoch, args.beta, args.anneal_epochs)

        train_loss, train_recon, train_kl = train_one_epoch(model, train_loader, optimizer, DEVICE, beta)
        val_loss, val_recon, val_kl = validate(model, val_loader, DEVICE, beta)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["recon_loss"].append(train_recon)
        history["kl_loss"].append(train_kl)
        history["beta"].append(beta)

        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {train_loss:.4f} | "
                  f"Recon: {train_recon:.4f} | KL: {train_kl:.4f} | β: {beta:.4f} | ValLoss: {val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vae_best.pt"))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vae_final.pt"))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(os.path.join(PLOTS_DIR, "vae_training_history.json"), "w") as f:
        json.dump(history, f)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].set_title("Total VAE Loss")
        axes[0].legend(); axes[0].grid(True)

        axes[1].plot(history["recon_loss"], label="Recon Loss", color="blue")
        axes[1].set_title("Reconstruction Loss")
        axes[1].legend(); axes[1].grid(True)

        axes[2].plot(history["kl_loss"], label="KL Divergence", color="red")
        axes[2].plot(history["beta"], label="β (annealing)", color="green", linestyle="--")
        axes[2].set_title("KL Divergence & β Schedule")
        axes[2].legend(); axes[2].grid(True)

        for ax in axes:
            ax.set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "vae_loss_curves.png"), dpi=150)
        plt.close()
        print(f"Loss curves saved to {PLOTS_DIR}/vae_loss_curves.png")
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")

    # --- Generate 8 multi-genre samples ---
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae_best.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.eval()

    os.makedirs(GENERATED_MIDI_DIR, exist_ok=True)
    from src.generation.midi_export import piano_roll_to_midi

    for i, genre_name in enumerate(GENRES):
        genre_tensor = torch.tensor([i], dtype=torch.long).to(DEVICE)
        generated = model.generate(num_samples=1, genre=genre_tensor, device=DEVICE)
        roll = generated[0].cpu().numpy()
        midi_path = os.path.join(GENERATED_MIDI_DIR, f"vae_generated_{genre_name}.mid")
        piano_roll_to_midi(roll, midi_path, threshold=0.3)

    # Extra samples for total of 8
    for i in range(3):
        generated = model.generate(num_samples=1, device=DEVICE)
        roll = generated[0].cpu().numpy()
        midi_path = os.path.join(GENERATED_MIDI_DIR, f"vae_generated_random_{i+1}.mid")
        piano_roll_to_midi(roll, midi_path, threshold=0.3)

    print(f"Generated 8 multi-genre MIDI samples in {GENERATED_MIDI_DIR}")
    print("Task 2 training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train VAE (Task 2)")
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=VAE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VAE_LEARNING_RATE)
    parser.add_argument("--hidden_dim", type=int, default=VAE_HIDDEN_DIM)
    parser.add_argument("--latent_dim", type=int, default=VAE_LATENT_DIM)
    parser.add_argument("--num_layers", type=int, default=VAE_NUM_LAYERS)
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--dropout", type=float, default=VAE_DROPOUT)
    parser.add_argument("--beta", type=float, default=VAE_BETA)
    parser.add_argument("--anneal_epochs", type=int, default=VAE_BETA_ANNEAL_EPOCHS)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for CPU training")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
