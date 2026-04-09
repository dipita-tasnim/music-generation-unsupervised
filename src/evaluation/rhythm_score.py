"""
Rhythm analysis metrics for music evaluation.

Rhythm Diversity Score:
    D_rhythm = #unique_durations / #total_notes

Repetition Ratio:
    R = #repeated_patterns / #total_patterns
"""
import numpy as np
from collections import Counter


def rhythm_diversity_score(durations: list[float], quantize_resolution: float = 0.05) -> float:
    """Compute rhythm diversity score.

    D_rhythm = #unique_durations / #total_notes

    Args:
        durations: List of note durations in seconds
        quantize_resolution: Resolution for quantizing durations (to avoid float precision issues)

    Returns:
        Rhythm diversity score in [0, 1]. Higher = more diverse rhythm.
    """
    if not durations:
        return 0.0

    # Quantize durations to avoid floating-point noise
    quantized = [round(d / quantize_resolution) * quantize_resolution for d in durations]
    unique_count = len(set(quantized))
    total_count = len(quantized)

    return unique_count / total_count


def repetition_ratio(pitches: list[int], pattern_length: int = 4) -> float:
    """Compute repetition ratio of pitch patterns.

    R = #repeated_patterns / #total_patterns

    Args:
        pitches: List of MIDI pitch values
        pattern_length: Length of n-gram patterns to detect

    Returns:
        Repetition ratio in [0, 1]. Higher = more repetitive.
    """
    if len(pitches) < pattern_length:
        return 0.0

    # Extract all n-grams
    patterns = []
    for i in range(len(pitches) - pattern_length + 1):
        pattern = tuple(pitches[i:i + pattern_length])
        patterns.append(pattern)

    total_patterns = len(patterns)
    if total_patterns == 0:
        return 0.0

    pattern_counts = Counter(patterns)
    repeated = sum(count - 1 for count in pattern_counts.values() if count > 1)

    return repeated / total_patterns


def note_density(notes: list[dict], total_duration: float | None = None) -> float:
    """Compute note density (notes per second).

    Args:
        notes: List of note dicts
        total_duration: Total duration in seconds (auto-computed if None)

    Returns:
        Notes per second
    """
    if not notes:
        return 0.0

    if total_duration is None:
        total_duration = max(n["end"] for n in notes) - min(n["start"] for n in notes)

    if total_duration <= 0:
        return 0.0

    return len(notes) / total_duration


def rhythm_pattern_analysis(durations: list[float], quantize_resolution: float = 0.05) -> dict:
    """Analyze rhythm patterns in a piece.

    Returns:
        Dictionary with rhythm statistics
    """
    if not durations:
        return {"unique_durations": 0, "total_notes": 0, "diversity": 0.0, "most_common": []}

    quantized = [round(d / quantize_resolution) * quantize_resolution for d in durations]
    counter = Counter(quantized)

    return {
        "unique_durations": len(counter),
        "total_notes": len(quantized),
        "diversity": rhythm_diversity_score(durations, quantize_resolution),
        "most_common": counter.most_common(5),
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "min_duration": float(min(durations)),
        "max_duration": float(max(durations)),
    }


def plot_rhythm_distribution(durations: list[float], title: str = "Rhythm Distribution",
                             save_path: str | None = None, bins: int = 50):
    """Plot histogram of note durations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(durations, bins=bins, color="coral", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
