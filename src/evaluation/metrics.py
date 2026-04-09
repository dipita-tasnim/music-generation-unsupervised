"""
Evaluation metrics for multi-genre music generation.

Includes:
    - Pitch Histogram Similarity
    - Rhythm Diversity Score
    - Repetition Ratio
    - Human Listening Score aggregation
    - Perplexity (for Transformer)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.evaluation.pitch_histogram import pitch_histogram_similarity, compute_pitch_histogram
from src.evaluation.rhythm_score import rhythm_diversity_score, repetition_ratio


def evaluate_generated_notes(generated_notes: list[list[dict]],
                             reference_notes: list[list[dict]] | None = None) -> dict:
    """Compute all quantitative metrics for a set of generated music pieces.

    Args:
        generated_notes: List of note-sequences (each is a list of note dicts)
        reference_notes: Optional reference note-sequences for comparison

    Returns:
        Dictionary of metric name -> value
    """
    results = {}

    # --- Pitch Histogram Similarity ---
    if reference_notes:
        ref_hist = compute_pitch_histogram(
            [n for piece in reference_notes for n in piece]
        )
        gen_hist = compute_pitch_histogram(
            [n for piece in generated_notes for n in piece]
        )
        results["pitch_histogram_similarity"] = pitch_histogram_similarity(ref_hist, gen_hist)
    else:
        gen_hist = compute_pitch_histogram(
            [n for piece in generated_notes for n in piece]
        )
        results["pitch_histogram"] = gen_hist.tolist()

    # --- Rhythm Diversity ---
    all_durations = [n["duration"] for piece in generated_notes for n in piece]
    results["rhythm_diversity"] = rhythm_diversity_score(all_durations)

    # --- Repetition Ratio ---
    all_pitches = [[n["pitch"] for n in piece] for piece in generated_notes]
    rep_ratios = [repetition_ratio(pitches) for pitches in all_pitches]
    results["repetition_ratio_mean"] = float(np.mean(rep_ratios))
    results["repetition_ratio_std"] = float(np.std(rep_ratios))

    # --- Per-piece metrics ---
    per_piece = []
    for i, piece in enumerate(generated_notes):
        durations = [n["duration"] for n in piece]
        pitches = [n["pitch"] for n in piece]
        per_piece.append({
            "piece_index": i,
            "num_notes": len(piece),
            "pitch_range": int(max(pitches) - min(pitches)) if pitches else 0,
            "rhythm_diversity": rhythm_diversity_score(durations),
            "repetition_ratio": repetition_ratio(pitches),
        })
    results["per_piece"] = per_piece

    return results


def format_comparison_table(results_dict: dict[str, dict]) -> str:
    """Format a comparison table from multiple model results.

    Args:
        results_dict: {model_name: metrics_dict}
    """
    header = f"{'Model':<25} {'Loss':<10} {'PPL':<10} {'Rhythm Div':<12} {'Rep Ratio':<12} {'Human Score':<12}"
    separator = "-" * len(header)
    rows = [header, separator]

    for model_name, metrics in results_dict.items():
        loss = f"{metrics.get('loss', '–')}"
        ppl = f"{metrics.get('perplexity', '–')}"
        rhythm = f"{metrics.get('rhythm_diversity', 0):.4f}" if isinstance(metrics.get('rhythm_diversity'), float) else "–"
        rep = f"{metrics.get('repetition_ratio_mean', 0):.4f}" if isinstance(metrics.get('repetition_ratio_mean'), float) else "–"
        human = f"{metrics.get('human_score', '–')}"
        rows.append(f"{model_name:<25} {loss:<10} {ppl:<10} {rhythm:<12} {rep:<12} {human:<12}")

    return "\n".join(rows)


def main():
    """Run evaluation on all generated MIDI files."""
    from src.config import GENERATED_MIDI_DIR, PLOTS_DIR
    from src.preprocessing.midi_parser import parse_midi_file

    import glob

    # Collect generated MIDI files
    midi_files = sorted(glob.glob(os.path.join(GENERATED_MIDI_DIR, "*.mid")))
    if not midi_files:
        print("No generated MIDI files found. Run generation first.")
        return

    # Group by model type
    model_groups = {}
    for path in midi_files:
        fname = os.path.basename(path)
        if fname.startswith("ae_"):
            model_groups.setdefault("Task 1: Autoencoder", []).append(path)
        elif fname.startswith("vae_"):
            model_groups.setdefault("Task 2: VAE", []).append(path)
        elif fname.startswith("transformer_"):
            model_groups.setdefault("Task 3: Transformer", []).append(path)
        elif fname.startswith("rlhf_"):
            model_groups.setdefault("Task 4: RLHF", []).append(path)

    all_results = {}
    for model_name, paths in model_groups.items():
        notes_list = [parse_midi_file(p) for p in paths]
        notes_list = [n for n in notes_list if n]  # filter empty
        if notes_list:
            results = evaluate_generated_notes(notes_list)
            all_results[model_name] = results
            print(f"\n{model_name}:")
            print(f"  Rhythm Diversity: {results['rhythm_diversity']:.4f}")
            print(f"  Repetition Ratio: {results['repetition_ratio_mean']:.4f} ± {results['repetition_ratio_std']:.4f}")

    # Save results
    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, "evaluation_results.json")
    # Convert numpy types for JSON serialization
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            kk: (vv if not isinstance(vv, np.floating) else float(vv))
            for kk, vv in v.items()
        }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print("\n" + format_comparison_table(all_results))


if __name__ == "__main__":
    main()
