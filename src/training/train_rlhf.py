"""
Training script for Task 4: RLHF Fine-Tuning of VAE Generator.

Uses REINFORCE policy gradient with heuristic rewards to fine-tune
the pretrained VAE model for improved music quality.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    VAE_HIDDEN_DIM, VAE_LATENT_DIM, VAE_NUM_LAYERS, VAE_DROPOUT,
    NUM_PITCHES, SEQUENCE_LENGTH, NUM_GENRES, DEVICE, RANDOM_SEED,
    RLHF_LEARNING_RATE, RLHF_ITERATIONS, RLHF_BATCH_SIZE,
    CHECKPOINT_DIR, PLOTS_DIR, GENERATED_MIDI_DIR,
)
from src.models.vae import MusicVAE
from src.models.rlhf import RLHFTrainer, heuristic_reward, HumanSurveyManager
from src.generation.midi_export import piano_roll_to_midi


def train(args):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(GENERATED_MIDI_DIR, exist_ok=True)

    # --- Load pretrained VAE ---
    model = MusicVAE(
        input_dim=NUM_PITCHES,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        num_genres=NUM_GENRES,
        dropout=args.dropout,
    ).to(DEVICE)

    vae_checkpoint = os.path.join(CHECKPOINT_DIR, "vae_best.pt")
    if os.path.exists(vae_checkpoint):
        model.load_state_dict(torch.load(vae_checkpoint, map_location=DEVICE, weights_only=True))
        print(f"Loaded pretrained VAE from {vae_checkpoint}")
    else:
        print("[WARN] No pretrained VAE found. Training RLHF from scratch (results may be poor).")

    # --- Pre-RLHF generation for comparison ---
    print("Generating pre-RLHF samples for comparison...")
    model.eval()
    with torch.no_grad():
        pre_samples = model.generate(num_samples=10, device=DEVICE)

    pre_rewards = []
    for i in range(pre_samples.size(0)):
        roll = pre_samples[i].cpu().numpy()
        r = heuristic_reward(roll)
        pre_rewards.append(r)
    print(f"Pre-RLHF mean reward: {np.mean(pre_rewards):.3f}")

    # --- RLHF Training with early stopping ---
    trainer = RLHFTrainer(
        generator=model,
        reward_fn=heuristic_reward,
        lr=args.lr,
    )

    # Save pre-RLHF weights as fallback
    import copy
    best_weights = copy.deepcopy(model.state_dict())
    best_reward = np.mean(pre_rewards)

    print(f"Starting RLHF training for {args.iterations} iterations...")
    log_every = max(1, args.iterations // 10)
    patience_counter = 0
    patience_limit = 20  # stop if no improvement for 20 steps

    for step in range(1, args.iterations + 1):
        reward, loss = trainer.train_step_piano_roll(args.batch_size)

        if step % log_every == 0 or step == 1:
            print(f"Step {step:5d}/{args.iterations} | Reward: {reward:.3f} | Loss: {loss:.4f}", flush=True)

        # Periodically evaluate with more samples for stable reward estimate
        if step % 5 == 0:
            model.eval()
            with torch.no_grad():
                eval_samples = model.generate(num_samples=10, device=DEVICE)
            eval_rewards = [heuristic_reward(eval_samples[i].cpu().numpy()) for i in range(eval_samples.size(0))]
            eval_mean = np.mean(eval_rewards)
            model.train()

            if eval_mean > best_reward:
                best_reward = eval_mean
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"Early stopping at step {step} (no improvement for {patience_limit} eval checks)")
                break

    print("RLHF training complete.")

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"Restored best model (reward: {best_reward:.3f})")

    # --- Save model ---
    rlhf_path = os.path.join(CHECKPOINT_DIR, "vae_rlhf.pt")
    torch.save(model.state_dict(), rlhf_path)
    print(f"RLHF-tuned model saved to {rlhf_path}")

    # --- Post-RLHF generation ---
    print("Generating post-RLHF samples...")
    model.eval()
    with torch.no_grad():
        post_samples = model.generate(num_samples=5, device=DEVICE)

    post_rewards = []
    for i in range(post_samples.size(0)):
        roll = post_samples[i].cpu().numpy()
        r = heuristic_reward(roll)
        post_rewards.append(r)

        out_path = os.path.join(GENERATED_MIDI_DIR, f"rlhf_sample_{i+1}.mid")
        piano_roll_to_midi(roll, out_path, threshold=0.3)

    print(f"Post-RLHF mean reward: {np.mean(post_rewards):.3f}")
    print(f"Improvement: {np.mean(post_rewards) - np.mean(pre_rewards):.3f}")

    # --- Plot reward curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(trainer.history["reward"], alpha=0.7)
    ax1.set_xlabel("RLHF Iteration")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("RLHF Reward Curve")
    ax1.grid(True, alpha=0.3)

    ax2.plot(trainer.history["loss"], alpha=0.7, color="red")
    ax2.set_xlabel("RLHF Iteration")
    ax2.set_ylabel("Policy Gradient Loss")
    ax2.set_title("RLHF Loss Curve")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "rlhf_training.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"RLHF plots saved to {plot_path}")

    # --- Create survey template ---
    all_midis = [os.path.join(GENERATED_MIDI_DIR, f)
                 for f in os.listdir(GENERATED_MIDI_DIR) if f.endswith(".mid")]
    survey_mgr = HumanSurveyManager()
    survey_mgr.create_survey_template(all_midis)

    print("Task 4 RLHF training complete!")


def main():
    parser = argparse.ArgumentParser(description="Task 4: RLHF Fine-Tuning")
    parser.add_argument("--iterations", type=int, default=RLHF_ITERATIONS)
    parser.add_argument("--batch_size", type=int, default=RLHF_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=RLHF_LEARNING_RATE)
    parser.add_argument("--hidden_dim", type=int, default=VAE_HIDDEN_DIM)
    parser.add_argument("--latent_dim", type=int, default=VAE_LATENT_DIM)
    parser.add_argument("--num_layers", type=int, default=VAE_NUM_LAYERS)
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--dropout", type=float, default=VAE_DROPOUT)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
