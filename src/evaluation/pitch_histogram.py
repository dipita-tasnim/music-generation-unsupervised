"""
Pitch Histogram analysis for music evaluation.

Pitch Histogram Similarity:
    H(p, q) = sum_{i=1}^{12} |p_i - q_i|

where p and q are pitch-class distributions (12 pitch classes: C, C#, ..., B).
"""
import numpy as np


def compute_pitch_histogram(notes: list[dict], normalize: bool = True) -> np.ndarray:
    """Compute a 12-bin pitch-class histogram from note events.

    Args:
        notes: List of note dicts with 'pitch' key (MIDI 0-127)
        normalize: Whether to normalize to a probability distribution

    Returns:
        np.ndarray of shape (12,) — pitch class distribution
    """
    histogram = np.zeros(12, dtype=np.float64)
    for note in notes:
        pitch_class = note["pitch"] % 12
        histogram[pitch_class] += 1

    if normalize and histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def pitch_histogram_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Compute L1 pitch histogram similarity (lower = more similar).

    H(p, q) = sum_{i=1}^{12} |p_i - q_i|

    Args:
        p: Reference pitch-class distribution (12,)
        q: Generated pitch-class distribution (12,)

    Returns:
        L1 distance (0.0 = identical, 2.0 = maximally different)
    """
    return float(np.sum(np.abs(p - q)))


def pitch_class_entropy(histogram: np.ndarray) -> float:
    """Compute entropy of the pitch-class distribution (higher = more diverse)."""
    h = histogram[histogram > 0]
    return float(-np.sum(h * np.log2(h)))


def plot_pitch_histogram(histogram: np.ndarray, title: str = "Pitch Class Distribution",
                         save_path: str | None = None):
    """Plot a pitch-class histogram as a bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pitch_names, histogram, color="steelblue", edgecolor="black")
    ax.set_xlabel("Pitch Class")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.set_ylim(0, max(histogram) * 1.2 if max(histogram) > 0 else 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def compare_pitch_histograms(histograms: dict[str, np.ndarray],
                             save_path: str | None = None):
    """Plot multiple pitch histograms overlaid for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    x = np.arange(12)
    width = 0.8 / len(histograms)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, hist) in enumerate(histograms.items()):
        ax.bar(x + i * width, hist, width, label=name, alpha=0.8)

    ax.set_xticks(x + width * (len(histograms) - 1) / 2)
    ax.set_xticklabels(pitch_names)
    ax.set_xlabel("Pitch Class")
    ax.set_ylabel("Proportion")
    ax.set_title("Pitch Class Distribution Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
