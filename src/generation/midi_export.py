"""
MIDI Export — converts model outputs (piano-rolls or token sequences) to MIDI files.
"""
import os
import numpy as np
import pretty_midi

from src.config import PIANO_ROLL_FPS, NUM_PITCHES


def piano_roll_to_midi(roll: np.ndarray, output_path: str,
                       fps: int = PIANO_ROLL_FPS, velocity: int = 80,
                       threshold: float = 0.5, instrument_name: str = "Acoustic Grand Piano",
                       adaptive: bool = True, top_k: int = 4):
    """Convert a piano-roll matrix to a MIDI file.

    Args:
        roll: (T, 128) piano-roll matrix (values in [0, 1])
        output_path: Path to save the MIDI file
        fps: Frames per second of the piano roll
        velocity: Default MIDI velocity for notes
        threshold: Activation threshold for binarizing
        instrument_name: General MIDI instrument name
        adaptive: If True, use top-k per timestep when fixed threshold yields too few notes
        top_k: Number of pitches to keep per timestep in adaptive mode
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    # Binarize — use adaptive top-k when fixed threshold produces sparse output
    binary = (roll >= threshold).astype(np.uint8)
    if adaptive and binary.sum() < roll.shape[0]:
        # Fixed threshold gives < 1 note per timestep on average — use top-k
        binary = np.zeros_like(roll, dtype=np.uint8)
        for t in range(roll.shape[0]):
            row = roll[t]
            if row.max() > 0.01:  # skip silent timesteps
                top_indices = np.argsort(row)[-top_k:]
                for idx in top_indices:
                    if row[idx] > 0.01:  # minimal sanity threshold
                        binary[t, idx] = 1

    for pitch in range(min(roll.shape[1], NUM_PITCHES)):
        in_note = False
        start_step = 0

        for step in range(binary.shape[0]):
            if binary[step, pitch] and not in_note:
                in_note = True
                start_step = step
            elif not binary[step, pitch] and in_note:
                in_note = False
                start_time = start_step / fps
                end_time = step / fps
                if end_time > start_time:
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time,
                    )
                    instrument.notes.append(note)

        # Close note at the end
        if in_note:
            start_time = start_step / fps
            end_time = binary.shape[0] / fps
            if end_time > start_time:
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
                instrument.notes.append(note)

    pm.instruments.append(instrument)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm.write(output_path)


def tokens_to_midi(tokens: list[int], output_path: str,
                   instrument_name: str = "Acoustic Grand Piano"):
    """Convert a token sequence to a MIDI file.

    Args:
        tokens: Token sequence from the tokenizer
        output_path: Path to save the MIDI file
        instrument_name: General MIDI instrument name
    """
    from src.preprocessing.tokenizer import tokens_to_notes

    notes = tokens_to_notes(tokens)
    notes_to_midi(notes, output_path, instrument_name)


def notes_to_midi(notes: list[dict], output_path: str,
                  instrument_name: str = "Acoustic Grand Piano"):
    """Convert note dicts to a MIDI file.

    Args:
        notes: List of note dicts with pitch, start, end, velocity
        output_path: Path to save the MIDI file
        instrument_name: General MIDI instrument name
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    for note_dict in notes:
        if note_dict["end"] <= note_dict["start"]:
            continue
        note = pretty_midi.Note(
            velocity=min(127, max(1, note_dict.get("velocity", 80))),
            pitch=max(0, min(127, note_dict["pitch"])),
            start=max(0.0, note_dict["start"]),
            end=max(0.01, note_dict["end"]),
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm.write(output_path)
