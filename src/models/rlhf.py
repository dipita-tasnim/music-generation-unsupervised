"""
Task 4: Reinforcement Learning for Human Preference Tuning (RLHF).

Algorithm 4:
    1. Start with pretrained generator policy p_theta(X)
    2. For each iteration:
        Generate music samples: X_gen ~ p_theta(X)
        Collect human preference score: r = HumanScore(X_gen)
        Compute expected reward: J(theta) = E[r(X_gen)]
        Policy gradient update: grad_theta J(theta) = E[r * grad_theta log p_theta(X)]
        Update: theta <- theta + eta * grad_theta J(theta)
    3. Output RLHF-tuned music generation model
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    RLHF_LEARNING_RATE, RLHF_ITERATIONS, RLHF_BATCH_SIZE,
    RLHF_CLIP_EPSILON, DEVICE, CHECKPOINT_DIR, GENERATED_MIDI_DIR,
    PLOTS_DIR, SURVEY_DIR,
)


class RewardModel(nn.Module):
    """Simple reward model that scores music quality.

    In practice this would be trained on human preference data.
    For demonstration, we use a proxy based on musical heuristics:
    - Pitch variety (moderate is good)
    - Rhythm consistency
    - Note density
    - Avoid extreme repetition
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) piano-roll or feature vector
        Returns:
            reward: (batch, 1) scalar reward
        """
        # Global average pooling over time
        x_pooled = x.mean(dim=1)  # (batch, input_dim)
        return self.net(x_pooled)


def heuristic_reward(piano_roll: np.ndarray) -> float:
    """Compute a heuristic music quality reward (proxy for human preference).

    Scores:
        [0, 5] range mimicking human listening scores

    Components:
        - Pitch diversity: moderate number of unique pitches is good
        - Note density: not too sparse, not too dense
        - Rhythmic structure: some repetition but not too much
    """
    if piano_roll.shape[0] == 0:
        return 0.0

    binary = (piano_roll > 0.5).astype(np.float32)

    # Pitch diversity (how many unique pitches are active)
    active_pitches = np.where(binary.sum(axis=0) > 0)[0]
    num_active = len(active_pitches)
    # Ideal: 15-40 active pitches
    pitch_score = min(num_active / 15, 1.0) * (1 - max(0, (num_active - 40)) / 50)
    pitch_score = max(0, pitch_score)

    # Note density (average active notes per time step)
    density = binary.sum(axis=1).mean()
    # Ideal: 2-6 notes per step
    density_score = min(density / 2, 1.0) * (1 - max(0, (density - 6)) / 10)
    density_score = max(0, density_score)

    # Temporal variation (not constant)
    temporal_var = np.diff(binary.sum(axis=1)).std()
    variation_score = min(temporal_var / 2, 1.0)

    # Combine into [0, 5] range
    reward = (pitch_score + density_score + variation_score) / 3 * 5
    return float(np.clip(reward, 0, 5))


class RLHFTrainer:
    """RLHF trainer using REINFORCE (policy gradient) to fine-tune a generator.

    Objective: max_theta E[r(X_gen)]
    Gradient:  grad_theta J(theta) = E[r * grad_theta log p_theta(X)]
    """

    def __init__(self, generator, reward_fn=None, lr: float = RLHF_LEARNING_RATE,
                 clip_epsilon: float = RLHF_CLIP_EPSILON):
        self.generator = generator
        self.reward_fn = reward_fn or heuristic_reward
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.history = {"reward": [], "loss": []}

    def compute_policy_gradient_loss(self, x_gen: torch.Tensor,
                                     log_probs: torch.Tensor,
                                     rewards: torch.Tensor) -> torch.Tensor:
        """REINFORCE loss: -E[r * log p_theta(X)]"""
        # Normalize rewards (baseline subtraction)
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss = -(log_probs * rewards_normalized).mean()
        return loss

    def train_step_piano_roll(self, batch_size: int = RLHF_BATCH_SIZE):
        """One RLHF training step for piano-roll-based generators (AE/VAE)."""
        self.generator.train()

        # Generate samples
        generated = self.generator.generate(num_samples=batch_size, device=DEVICE)
        # (batch, seq_len, 128)

        # Compute rewards
        rewards = []
        for i in range(generated.size(0)):
            roll = generated[i].detach().cpu().numpy()
            r = self.reward_fn(roll)
            rewards.append(r)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

        # Approximate log-probability via reconstruction likelihood (normalized)
        x_hat, *extra = self.generator(generated)
        # Use mean instead of sum to keep log_probs in a reasonable range
        log_probs = -F.mse_loss(x_hat, generated, reduction="none").mean(dim=(1, 2))

        # Policy gradient
        self.optimizer.zero_grad()
        loss = self.compute_policy_gradient_loss(generated, log_probs, rewards)
        loss.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer.step()

        mean_reward = rewards.mean().item()
        self.history["reward"].append(mean_reward)
        self.history["loss"].append(loss.item())

        return mean_reward, loss.item()

    def train(self, num_iterations: int = RLHF_ITERATIONS,
              batch_size: int = RLHF_BATCH_SIZE, log_every: int = 50):
        """Full RLHF training loop."""
        print(f"Starting RLHF training for {num_iterations} iterations...")

        for step in range(1, num_iterations + 1):
            reward, loss = self.train_step_piano_roll(batch_size)

            if step % log_every == 0 or step == 1:
                print(f"Step {step:5d}/{num_iterations} | Reward: {reward:.3f} | Loss: {loss:.4f}")

        print("RLHF training complete.")
        return self.history

    def save_history(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f)


class HumanSurveyManager:
    """Utility for collecting and managing human listening survey data.

    Survey format: Each participant rates generated music on a [1, 5] scale.
    """

    def __init__(self, output_dir: str = SURVEY_DIR):
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)

    def create_survey_template(self, midi_files: list[str], num_participants: int = 10):
        """Create a survey template JSON file."""
        survey = {
            "instructions": (
                "Please listen to each music sample and rate it on a scale of 1-5:\n"
                "  1 = Very poor (random/noisy)\n"
                "  2 = Poor (some structure but unpleasant)\n"
                "  3 = Fair (recognizable as music)\n"
                "  4 = Good (pleasant and coherent)\n"
                "  5 = Excellent (creative and musical)"
            ),
            "samples": [
                {
                    "id": i + 1,
                    "file": os.path.basename(f),
                    "model": self._infer_model(f),
                    "ratings": {f"participant_{j+1}": None for j in range(num_participants)},
                }
                for i, f in enumerate(midi_files)
            ],
        }
        survey_path = os.path.join(self.output_dir, "human_survey_template.json")
        with open(survey_path, "w") as f:
            json.dump(survey, f, indent=2)
        print(f"Survey template saved to {survey_path}")
        return survey_path

    @staticmethod
    def _infer_model(filepath: str) -> str:
        fname = os.path.basename(filepath).lower()
        if "random" in fname or "baseline_random" in fname:
            return "Random Generator"
        elif "markov" in fname:
            return "Markov Chain"
        elif "ae_" in fname:
            return "LSTM Autoencoder"
        elif "vae_" in fname:
            return "VAE"
        elif "transformer" in fname:
            return "Transformer"
        elif "rlhf" in fname:
            return "RLHF-Tuned"
        return "Unknown"

    def load_survey_results(self, survey_path: str) -> dict:
        """Load completed survey results and compute statistics."""
        with open(survey_path, "r") as f:
            survey = json.load(f)

        model_scores = {}
        for sample in survey["samples"]:
            model = sample["model"]
            ratings = [v for v in sample["ratings"].values() if v is not None]
            if ratings:
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].extend(ratings)

        stats = {}
        for model, scores in model_scores.items():
            stats[model] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(min(scores)),
                "max": float(max(scores)),
                "n_ratings": len(scores),
            }

        return stats

    def plot_survey_results(self, stats: dict, save_path: str | None = None):
        """Plot human listening survey results as a bar chart."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(stats.keys())
        means = [stats[m]["mean"] for m in models]
        stds = [stats[m]["std"] for m in models]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models, means, yerr=stds, capsize=5,
                       color=["gray", "lightblue", "blue", "green", "orange", "red"][:len(models)],
                       edgecolor="black")
        ax.set_ylabel("Human Score [1-5]")
        ax.set_title("Human Listening Survey Results")
        ax.set_ylim(0, 5.5)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()
