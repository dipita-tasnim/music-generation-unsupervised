"""
Piano Roll — converts parsed MIDI note data to piano-roll matrix representation.
Used as an alternative input format for convolutional or recurrent models.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    NUM_PITCHES, PIANO_ROLL_FPS, SEQUENCE_LENGTH, MIDI_RESOLUTION,
)


def notes_to_piano_roll(notes: list[dict], fps: int = PIANO_ROLL_FPS,
                        max_time: float | None = None) -> np.ndarray:
    """Convert note events to a binary piano-roll matrix.

    Returns:
        np.ndarray of shape (num_time_steps, 128), dtype float32.
    """
    if not notes:
        return np.zeros((1, NUM_PITCHES), dtype=np.float32)

    if max_time is None:
        max_time = max(n["end"] for n in notes)

    num_steps = int(np.ceil(max_time * fps))
    roll = np.zeros((num_steps, NUM_PITCHES), dtype=np.float32)

    for note in notes:
        start_step = int(round(note["start"] * fps))
        end_step = int(round(note["end"] * fps))
        start_step = max(0, min(start_step, num_steps - 1))
        end_step = max(start_step + 1, min(end_step, num_steps))
        pitch = note["pitch"]
        if 0 <= pitch < NUM_PITCHES:
            roll[start_step:end_step, pitch] = 1.0

    return roll


def notes_to_velocity_roll(notes: list[dict], fps: int = PIANO_ROLL_FPS,
                           max_time: float | None = None) -> np.ndarray:
    """Piano roll with velocity values (0.0–1.0) instead of binary."""
    if not notes:
        return np.zeros((1, NUM_PITCHES), dtype=np.float32)

    if max_time is None:
        max_time = max(n["end"] for n in notes)

    num_steps = int(np.ceil(max_time * fps))
    roll = np.zeros((num_steps, NUM_PITCHES), dtype=np.float32)

    for note in notes:
        start_step = int(round(note["start"] * fps))
        end_step = int(round(note["end"] * fps))
        start_step = max(0, min(start_step, num_steps - 1))
        end_step = max(start_step + 1, min(end_step, num_steps))
        pitch = note["pitch"]
        if 0 <= pitch < NUM_PITCHES:
            roll[start_step:end_step, pitch] = note["velocity"] / 127.0

    return roll


def piano_roll_to_notes(roll: np.ndarray, fps: int = PIANO_ROLL_FPS,
                        velocity: int = 80, threshold: float = 0.5) -> list[dict]:
    """Convert a piano-roll matrix back into note events."""
    notes = []
    active = {}  # pitch -> start_step

    for step in range(roll.shape[0]):
        for pitch in range(roll.shape[1]):
            is_on = roll[step, pitch] >= threshold
            if is_on and pitch not in active:
                active[pitch] = step
            elif not is_on and pitch in active:
                start_step = active.pop(pitch)
                notes.append({
                    "pitch": pitch,
                    "start": start_step / fps,
                    "end": step / fps,
                    "duration": (step - start_step) / fps,
                    "velocity": velocity,
                })

    # Close remaining active notes
    for pitch, start_step in active.items():
        notes.append({
            "pitch": pitch,
            "start": start_step / fps,
            "end": roll.shape[0] / fps,
            "duration": (roll.shape[0] - start_step) / fps,
            "velocity": velocity,
        })

    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def segment_piano_roll(roll: np.ndarray, seg_len: int = SEQUENCE_LENGTH,
                       stride: int | None = None) -> list[np.ndarray]:
    """Segment a piano roll into fixed-length windows."""
    if stride is None:
        stride = seg_len // 2

    segments = []
    T = roll.shape[0]
    for i in range(0, T - seg_len + 1, stride):
        segments.append(roll[i:i + seg_len])

    # Pad and add final segment if needed
    if T >= seg_len and (T - seg_len) % stride != 0:
        last = roll[-seg_len:]
        segments.append(last)
    elif T < seg_len:
        padded = np.zeros((seg_len, roll.shape[1]), dtype=roll.dtype)
        padded[:T] = roll
        segments.append(padded)

    return segments


class PianoRollDataset(Dataset):
    """PyTorch Dataset for piano-roll segments."""

    def __init__(self, segments: list[np.ndarray], genre_ids: list[int] | None = None):
        self.segments = [torch.tensor(s, dtype=torch.float32) for s in segments]
        self.genre_ids = genre_ids

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        item = {"piano_roll": self.segments[idx]}
        if self.genre_ids is not None:
            item["genre"] = self.genre_ids[idx]
        return item


def build_piano_roll_dataset(parsed_data: list[dict],
                             seg_len: int = SEQUENCE_LENGTH,
                             use_velocity: bool = False) -> PianoRollDataset:
    """Build a piano-roll dataset from parsed MIDI records."""
    from src.config import GENRE_TO_ID

    all_segments = []
    all_genres = []

    for record in parsed_data:
        if use_velocity:
            roll = notes_to_velocity_roll(record["notes"])
        else:
            roll = notes_to_piano_roll(record["notes"])

        segs = segment_piano_roll(roll, seg_len=seg_len)
        genre_id = GENRE_TO_ID.get(record.get("genre", "unknown"), -1)
        all_segments.extend(segs)
        all_genres.extend([genre_id] * len(segs))

    return PianoRollDataset(all_segments, all_genres)
