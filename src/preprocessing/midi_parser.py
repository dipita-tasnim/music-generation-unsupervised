"""
MIDI Parser — reads raw MIDI files and extracts note events.
Converts MIDI into structured note sequences for downstream processing.
"""
import os
import glob
import argparse
import json
import numpy as np
import pretty_midi


def parse_midi_file(midi_path: str) -> list[dict]:
    """Parse a single MIDI file into a list of note event dicts."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append({
                "pitch": note.pitch,
                "start": float(note.start),
                "end": float(note.end),
                "duration": float(note.end - note.start),
                "velocity": note.velocity,
            })
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def parse_midi_directory(input_dir: str, genre_label: str | None = None) -> list[dict]:
    """Parse all MIDI files in a directory, returning a list of song records."""
    midi_paths = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.mid"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**", "*.midi"), recursive=True)
    )

    dataset = []
    for path in midi_paths:
        notes = parse_midi_file(path)
        if not notes:
            continue
        record = {
            "filename": os.path.basename(path),
            "path": path,
            "genre": genre_label or infer_genre(path),
            "num_notes": len(notes),
            "duration_sec": max(n["end"] for n in notes),
            "notes": notes,
        }
        dataset.append(record)
    return dataset


def infer_genre(path: str) -> str:
    """Heuristic genre inference from directory structure."""
    from src.config import GENRES
    lower = path.lower()
    for genre in GENRES:
        if genre in lower:
            return genre
    return "unknown"


def extract_tempo_and_time_signature(midi_path: str):
    """Extract tempo changes and time signatures from a MIDI file."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    tempos = pm.get_tempo_changes()
    tempo_list = list(zip(tempos[0].tolist(), tempos[1].tolist())) if len(tempos[0]) > 0 else [(0.0, 120.0)]

    time_sigs = []
    for ts in pm.time_signature_changes:
        time_sigs.append({
            "time": ts.time,
            "numerator": ts.numerator,
            "denominator": ts.denominator,
        })
    if not time_sigs:
        time_sigs = [{"time": 0.0, "numerator": 4, "denominator": 4}]
    return tempo_list, time_sigs


def save_parsed_data(dataset: list[dict], output_path: str):
    """Save parsed dataset to a JSON file (notes stored as lists for efficiency)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert to serializable format
    serializable = []
    for record in dataset:
        r = {k: v for k, v in record.items() if k != "notes"}
        r["notes"] = record["notes"]
        serializable.append(r)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_parsed_data(json_path: str) -> list[dict]:
    """Load previously parsed dataset from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Parse raw MIDI files.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw MIDI directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save parsed JSON")
    parser.add_argument("--genre", type=str, default=None, help="Genre label for all files")
    args = parser.parse_args()

    dataset = parse_midi_directory(args.input, genre_label=args.genre)
    out_path = os.path.join(args.output, "parsed_midi.json")
    save_parsed_data(dataset, out_path)
    print(f"Parsed {len(dataset)} MIDI files → {out_path}")


if __name__ == "__main__":
    main()
