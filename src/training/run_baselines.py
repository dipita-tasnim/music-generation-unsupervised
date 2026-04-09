"""
Run baseline models (Random Generator & Markov Chain) for comparison.
Extracted from notebooks/baseline_markov.ipynb for CLI execution.
"""
import os
import sys
import json
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    GENERATED_MIDI_DIR, PLOTS_DIR, PROCESSED_DIR,
    NUM_PITCHES, RANDOM_SEED,
)
from src.preprocessing.midi_parser import load_parsed_data
from src.generation.midi_export import notes_to_midi
from src.evaluation.pitch_histogram import compute_pitch_histogram, pitch_histogram_similarity
from src.evaluation.rhythm_score import rhythm_diversity_score, repetition_ratio

np.random.seed(RANDOM_SEED)


def random_note_generator(num_notes=200, pitch_range=(40, 85),
                          duration_choices=(0.25, 0.5, 1.0, 1.5),
                          velocity_range=(50, 110)):
    notes = []
    current_time = 0.0
    for _ in range(num_notes):
        pitch = np.random.randint(pitch_range[0], pitch_range[1])
        duration = np.random.choice(duration_choices)
        velocity = np.random.randint(velocity_range[0], velocity_range[1])
        notes.append({
            'pitch': int(pitch),
            'start': float(current_time),
            'end': float(current_time + duration),
            'duration': float(duration),
            'velocity': int(velocity),
        })
        current_time += duration * np.random.choice([0.5, 1.0])
    return notes


class MarkovChainMusicModel:
    def __init__(self, order=1):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.duration_transitions = defaultdict(Counter)

    def fit(self, notes_list):
        pitches = [n['pitch'] for n in notes_list]
        durations = [round(n['duration'] * 4) / 4 for n in notes_list]
        for i in range(len(pitches) - self.order):
            state = tuple(pitches[i:i + self.order])
            self.transitions[state][pitches[i + self.order]] += 1
            dur_state = tuple(durations[i:i + self.order])
            self.duration_transitions[dur_state][durations[i + self.order]] += 1
        print(f'Learned {len(self.transitions)} pitch transitions, '
              f'{len(self.duration_transitions)} duration transitions')

    def generate(self, num_notes=200):
        if not self.transitions:
            raise ValueError('Model not fitted yet.')
        state = list(self.transitions.keys())[np.random.randint(len(self.transitions))]
        pitches = list(state)
        dur_state = list(self.duration_transitions.keys())[0]
        durations = list(dur_state)

        for _ in range(num_notes - self.order):
            state = tuple(pitches[-self.order:])
            if state in self.transitions:
                counter = self.transitions[state]
                items = list(counter.keys())
                weights = np.array(list(counter.values()), dtype=np.float64)
                weights /= weights.sum()
                next_pitch = np.random.choice(items, p=weights)
            else:
                next_pitch = np.random.randint(40, 85)
            pitches.append(int(next_pitch))

            dur_state = tuple(durations[-self.order:])
            if dur_state in self.duration_transitions:
                counter = self.duration_transitions[dur_state]
                items = list(counter.keys())
                weights = np.array(list(counter.values()), dtype=np.float64)
                weights /= weights.sum()
                next_dur = np.random.choice(items, p=weights)
            else:
                next_dur = np.random.choice([0.25, 0.5, 1.0])
            durations.append(float(next_dur))

        notes = []
        current_time = 0.0
        for pitch, duration in zip(pitches, durations):
            notes.append({
                'pitch': pitch, 'start': current_time,
                'end': current_time + duration, 'duration': duration,
                'velocity': np.random.randint(60, 100),
            })
            current_time += duration
        return notes


def main():
    os.makedirs(GENERATED_MIDI_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load reference data
    parsed_path = os.path.join(PROCESSED_DIR, 'parsed_midi.json')
    if os.path.exists(parsed_path):
        parsed_data = load_parsed_data(parsed_path)
        all_notes = [n for record in parsed_data for n in record['notes']]
        print(f'Loaded {len(parsed_data)} pieces, {len(all_notes):,} total notes')
    else:
        print('No parsed data found. Creating synthetic reference data.')
        all_notes = []
        for i in range(1000):
            all_notes.append({
                'pitch': int(np.random.choice([60, 62, 64, 65, 67, 69, 71, 72])),
                'start': i * 0.25,
                'end': i * 0.25 + float(np.random.choice([0.25, 0.5, 1.0])),
                'duration': float(np.random.choice([0.25, 0.5, 1.0])),
                'velocity': int(np.random.randint(60, 100)),
            })

    ref_histogram = compute_pitch_histogram(all_notes)

    # --- Random Generator ---
    print("\n=== Baseline 1: Random Note Generator ===")
    random_pieces = [random_note_generator(200) for _ in range(5)]
    for i, notes in enumerate(random_pieces):
        notes_to_midi(notes, os.path.join(GENERATED_MIDI_DIR, f'baseline_random_{i+1}.mid'))
    print(f'Generated 5 random baseline MIDI files')

    random_all_notes = [n for piece in random_pieces for n in piece]
    random_hist = compute_pitch_histogram(random_all_notes)
    random_durations = [n['duration'] for n in random_all_notes]
    random_pitches = [[n['pitch'] for n in piece] for piece in random_pieces]

    random_phs = pitch_histogram_similarity(ref_histogram, random_hist)
    random_rd = rhythm_diversity_score(random_durations)
    random_rr = float(np.mean([repetition_ratio(p) for p in random_pitches]))

    print(f'Pitch Histogram Similarity: {random_phs:.4f}')
    print(f'Rhythm Diversity: {random_rd:.4f}')
    print(f'Repetition Ratio: {random_rr:.4f}')

    # --- Markov Chain ---
    print("\n=== Baseline 2: Markov Chain ===")
    markov = MarkovChainMusicModel(order=1)
    markov.fit(all_notes)

    markov_pieces = [markov.generate(200) for _ in range(5)]
    for i, notes in enumerate(markov_pieces):
        notes_to_midi(notes, os.path.join(GENERATED_MIDI_DIR, f'baseline_markov_{i+1}.mid'))
    print(f'Generated 5 Markov chain baseline MIDI files')

    markov_all_notes = [n for piece in markov_pieces for n in piece]
    markov_hist = compute_pitch_histogram(markov_all_notes)
    markov_durations = [n['duration'] for n in markov_all_notes]
    markov_pitches = [[n['pitch'] for n in piece] for piece in markov_pieces]

    markov_phs = pitch_histogram_similarity(ref_histogram, markov_hist)
    markov_rd = rhythm_diversity_score(markov_durations)
    markov_rr = float(np.mean([repetition_ratio(p) for p in markov_pitches]))

    print(f'Pitch Histogram Similarity: {markov_phs:.4f}')
    print(f'Rhythm Diversity: {markov_rd:.4f}')
    print(f'Repetition Ratio: {markov_rr:.4f}')

    # --- Summary ---
    print("\n=== Baseline Comparison ===")
    print(f"{'Model':<22} {'Pitch Sim':>10} {'Rhythm Div':>12} {'Rep Ratio':>12}")
    print("-" * 58)
    print(f"{'Random Generator':<22} {random_phs:>10.4f} {random_rd:>12.4f} {random_rr:>12.4f}")
    print(f"{'Markov Chain':<22} {markov_phs:>10.4f} {markov_rd:>12.4f} {markov_rr:>12.4f}")

    # Save
    results = {
        'Random Generator': {
            'pitch_histogram_similarity': random_phs,
            'rhythm_diversity': random_rd,
            'repetition_ratio': random_rr,
        },
        'Markov Chain': {
            'pitch_histogram_similarity': markov_phs,
            'rhythm_diversity': markov_rd,
            'repetition_ratio': markov_rr,
        },
    }
    with open(os.path.join(PLOTS_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {PLOTS_DIR}/baseline_results.json')
    print("Baselines complete!")


if __name__ == "__main__":
    main()
