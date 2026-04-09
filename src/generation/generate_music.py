"""
High-level music generation script — unified interface for all models.

Usage:
    python -m src.generation.generate_music --model ae --num_samples 5
    python -m src.generation.generate_music --model vae --num_samples 8 --genre classical
    python -m src.generation.generate_music --model transformer --num_samples 10
"""
import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    DEVICE, CHECKPOINT_DIR, GENERATED_MIDI_DIR,
    AE_LATENT_DIM, AE_HIDDEN_DIM, AE_NUM_LAYERS, AE_DROPOUT,
    VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_NUM_LAYERS, VAE_DROPOUT,
    VOCAB_SIZE, TF_D_MODEL, TF_N_HEADS, TF_NUM_LAYERS, TF_D_FF, TF_DROPOUT,
    NUM_PITCHES, SEQUENCE_LENGTH, NUM_GENRES,
    BOS_TOKEN, GENRES, GENRE_TO_ID, TF_TEMPERATURE, TF_TOP_K, VAE_BETA,
)


def generate_ae(num_samples: int, output_dir: str):
    """Generate music using the trained LSTM Autoencoder."""
    from src.models.autoencoder import LSTMAutoencoder
    from src.generation.midi_export import piano_roll_to_midi

    model = LSTMAutoencoder(
        input_dim=NUM_PITCHES, hidden_dim=AE_HIDDEN_DIM,
        latent_dim=AE_LATENT_DIM, num_layers=AE_NUM_LAYERS,
        seq_len=SEQUENCE_LENGTH, dropout=AE_DROPOUT,
    ).to(DEVICE)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "autoencoder_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] No checkpoint found at {ckpt_path}. Using random weights.")

    model.eval()
    generated = model.generate(num_samples=num_samples, device=DEVICE)

    paths = []
    for i in range(generated.size(0)):
        roll = generated[i].cpu().numpy()
        midi_path = os.path.join(output_dir, f"ae_generated_{i+1}.mid")
        piano_roll_to_midi(roll, midi_path)
        paths.append(midi_path)

    print(f"Generated {num_samples} AE samples → {output_dir}")
    return paths


def generate_vae(num_samples: int, output_dir: str, genre: str | None = None):
    """Generate music using the trained VAE."""
    from src.models.vae import MusicVAE
    from src.generation.midi_export import piano_roll_to_midi

    model = MusicVAE(
        input_dim=NUM_PITCHES, hidden_dim=VAE_HIDDEN_DIM,
        latent_dim=VAE_LATENT_DIM, num_layers=VAE_NUM_LAYERS,
        seq_len=SEQUENCE_LENGTH, dropout=VAE_DROPOUT,
        num_genres=NUM_GENRES, beta=VAE_BETA,
    ).to(DEVICE)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "vae_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] No checkpoint found at {ckpt_path}. Using random weights.")

    model.eval()

    genre_tensor = None
    if genre and genre in GENRE_TO_ID:
        genre_tensor = torch.tensor([GENRE_TO_ID[genre]] * num_samples, dtype=torch.long).to(DEVICE)

    generated = model.generate(num_samples=num_samples, genre=genre_tensor, device=DEVICE)

    paths = []
    for i in range(generated.size(0)):
        roll = generated[i].cpu().numpy()
        suffix = f"_{genre}" if genre else f"_{i+1}"
        midi_path = os.path.join(output_dir, f"vae_generated{suffix}_{i+1}.mid")
        piano_roll_to_midi(roll, midi_path)
        paths.append(midi_path)

    print(f"Generated {num_samples} VAE samples → {output_dir}")
    return paths


def generate_transformer(num_samples: int, output_dir: str, genre: str | None = None,
                         max_len: int = 512, temperature: float = TF_TEMPERATURE,
                         top_k: int = TF_TOP_K):
    """Generate music using the trained Transformer."""
    from src.models.transformer import MusicTransformer
    from src.generation.midi_export import tokens_to_midi

    model = MusicTransformer(
        vocab_size=VOCAB_SIZE, d_model=TF_D_MODEL,
        n_heads=TF_N_HEADS, num_layers=TF_NUM_LAYERS,
        d_ff=TF_D_FF, max_seq_len=max_len,
        num_genres=NUM_GENRES, dropout=TF_DROPOUT,
    ).to(DEVICE)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "transformer_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] No checkpoint found at {ckpt_path}. Using random weights.")

    model.eval()

    paths = []
    for i in range(num_samples):
        genre_tensor = None
        genre_name = "unconditioned"
        if genre and genre in GENRE_TO_ID:
            genre_tensor = torch.tensor([GENRE_TO_ID[genre]], dtype=torch.long).to(DEVICE)
            genre_name = genre
        elif genre is None:
            genre_idx = i % NUM_GENRES
            genre_tensor = torch.tensor([genre_idx], dtype=torch.long).to(DEVICE)
            genre_name = GENRES[genre_idx]

        prompt = torch.tensor([[BOS_TOKEN]], dtype=torch.long).to(DEVICE)
        generated_tokens = model.generate(prompt, max_len=max_len, genre=genre_tensor,
                                          temperature=temperature, top_k=top_k)
        token_list = generated_tokens[0].cpu().tolist()
        midi_path = os.path.join(output_dir, f"transformer_generated_{genre_name}_{i+1}.mid")
        tokens_to_midi(token_list, midi_path)
        paths.append(midi_path)

    print(f"Generated {num_samples} Transformer samples → {output_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate music with trained models")
    parser.add_argument("--model", type=str, required=True, choices=["ae", "vae", "transformer"],
                        help="Which model to use for generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--genre", type=str, default=None, help="Genre for conditioned generation")
    parser.add_argument("--output_dir", type=str, default=GENERATED_MIDI_DIR)
    parser.add_argument("--max_len", type=int, default=512, help="Max length for Transformer")
    parser.add_argument("--temperature", type=float, default=TF_TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=TF_TOP_K)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "ae":
        generate_ae(args.num_samples, args.output_dir)
    elif args.model == "vae":
        generate_vae(args.num_samples, args.output_dir, args.genre)
    elif args.model == "transformer":
        generate_transformer(args.num_samples, args.output_dir, args.genre,
                             args.max_len, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
