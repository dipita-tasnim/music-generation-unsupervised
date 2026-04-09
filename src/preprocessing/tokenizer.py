"""
Tokenizer — converts parsed note events into token sequences for model consumption.
Supports note-on, note-off, velocity, and time-shift tokens.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    VOCAB_SIZE, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
    NOTE_ON_OFFSET, NOTE_OFF_OFFSET, VELOCITY_OFFSET,
    TIME_SHIFT_OFFSET, NUM_VELOCITIES, SEQUENCE_LENGTH,
    MAX_SEQUENCE_LENGTH, MIDI_RESOLUTION,
)


def quantize_velocity(velocity: int, num_bins: int = NUM_VELOCITIES) -> int:
    """Quantize MIDI velocity (0-127) into a fixed number of bins."""
    return min(int(velocity / 128 * num_bins), num_bins - 1)


def quantize_time(seconds: float, resolution: int = MIDI_RESOLUTION, tempo_bpm: float = 120.0) -> int:
    """Quantize a time interval in seconds to discrete time steps."""
    beats = seconds * tempo_bpm / 60.0
    steps = int(round(beats * resolution / 4))  # steps per beat = resolution/4
    return max(0, min(steps, 99))  # clamp to 100 possible time-shift values


def notes_to_tokens(notes: list[dict], tempo_bpm: float = 120.0) -> list[int]:
    """Convert a list of note dicts to a sequence of tokens.

    Token layout:
        [BOS] [time_shift] [note_on] [velocity] ... [note_off] ... [EOS]
    """
    if not notes:
        return [BOS_TOKEN, EOS_TOKEN]

    tokens = [BOS_TOKEN]
    prev_time = 0.0

    for note in notes:
        # Time shift from previous event
        dt = note["start"] - prev_time
        if dt > 0:
            steps = quantize_time(dt, tempo_bpm=tempo_bpm)
            if steps > 0:
                tokens.append(TIME_SHIFT_OFFSET + steps)

        # Note-on + velocity
        tokens.append(NOTE_ON_OFFSET + note["pitch"])
        vel_bin = quantize_velocity(note["velocity"])
        tokens.append(VELOCITY_OFFSET + vel_bin)

        # Note-off after duration
        dur_steps = quantize_time(note["duration"], tempo_bpm=tempo_bpm)
        if dur_steps > 0:
            tokens.append(TIME_SHIFT_OFFSET + dur_steps)
        tokens.append(NOTE_OFF_OFFSET + note["pitch"])

        prev_time = note["start"]

    tokens.append(EOS_TOKEN)
    return tokens


def tokens_to_notes(tokens: list[int], tempo_bpm: float = 120.0) -> list[dict]:
    """Decode a token sequence back into note dicts."""
    notes = []
    current_time = 0.0
    active_notes = {}  # pitch -> (start_time, velocity)

    for tok in tokens:
        if tok == PAD_TOKEN or tok == BOS_TOKEN or tok == EOS_TOKEN:
            continue
        elif TIME_SHIFT_OFFSET <= tok < TIME_SHIFT_OFFSET + 100:
            steps = tok - TIME_SHIFT_OFFSET
            beats = steps * 4 / MIDI_RESOLUTION
            current_time += beats * 60.0 / tempo_bpm
        elif NOTE_ON_OFFSET <= tok < NOTE_ON_OFFSET + 128:
            pitch = tok - NOTE_ON_OFFSET
            active_notes[pitch] = (current_time, 80)  # default velocity
        elif VELOCITY_OFFSET <= tok < VELOCITY_OFFSET + NUM_VELOCITIES:
            vel_bin = tok - VELOCITY_OFFSET
            velocity = int((vel_bin + 0.5) / NUM_VELOCITIES * 127)
            # Update last active note's velocity
            for pitch in list(active_notes.keys()):
                start, _ = active_notes[pitch]
                if start == current_time:
                    active_notes[pitch] = (start, velocity)
        elif NOTE_OFF_OFFSET <= tok < NOTE_OFF_OFFSET + 128:
            pitch = tok - NOTE_OFF_OFFSET
            if pitch in active_notes:
                start, velocity = active_notes.pop(pitch)
                notes.append({
                    "pitch": pitch,
                    "start": start,
                    "end": current_time,
                    "duration": current_time - start,
                    "velocity": velocity,
                })

    # Close any still-active notes
    for pitch, (start, velocity) in active_notes.items():
        notes.append({
            "pitch": pitch,
            "start": start,
            "end": current_time + 0.5,
            "duration": 0.5,
            "velocity": velocity,
        })

    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def segment_tokens(tokens: list[int], seq_len: int = SEQUENCE_LENGTH,
                   stride: int | None = None) -> list[list[int]]:
    """Segment a long token sequence into fixed-length overlapping windows."""
    if stride is None:
        stride = seq_len // 2

    segments = []
    for i in range(0, len(tokens) - seq_len + 1, stride):
        segments.append(tokens[i:i + seq_len])

    # Handle last segment with padding if needed
    if len(tokens) % stride != 0 or not segments:
        last = tokens[-(seq_len):]  if len(tokens) >= seq_len else tokens
        if len(last) < seq_len:
            last = last + [PAD_TOKEN] * (seq_len - len(last))
        if not segments or last != segments[-1]:
            segments.append(last)

    return segments


class MusicTokenDataset(Dataset):
    """PyTorch Dataset for tokenized music sequences."""

    def __init__(self, sequences: list[list[int]], genre_ids: list[int] | None = None):
        self.sequences = [torch.tensor(s, dtype=torch.long) for s in sequences]
        self.genre_ids = genre_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {"tokens": self.sequences[idx]}
        if self.genre_ids is not None:
            item["genre"] = self.genre_ids[idx]
        return item


def build_dataset_from_parsed(parsed_data: list[dict],
                              seq_len: int = SEQUENCE_LENGTH) -> MusicTokenDataset:
    """Build a tokenized dataset from parsed MIDI records."""
    from src.config import GENRE_TO_ID

    all_segments = []
    all_genres = []

    for record in parsed_data:
        tokens = notes_to_tokens(record["notes"])
        segments = segment_tokens(tokens, seq_len=seq_len)
        genre_id = GENRE_TO_ID.get(record.get("genre", "unknown"), -1)
        all_segments.extend(segments)
        all_genres.extend([genre_id] * len(segments))

    return MusicTokenDataset(all_segments, all_genres)
