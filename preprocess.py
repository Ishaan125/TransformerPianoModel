import os
import numpy as np
import pretty_midi
from tqdm import tqdm

def midi_to_sequence(file_path):
    """Convert a single MIDI file into lists of note pitches, velocities and durations.

    Returns:
        notes (list[int]): MIDI pitch values
        velocities (list[int]): note velocities
        durations (list[float]): note durations in seconds (end - start)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        notes = []
        velocities = []
        durations = []
        # iterate all instruments and collect non-drum notes
        for note in midi_data.instruments[0].notes:
            notes.append(note.pitch)
            velocities.append(note.velocity)
            durations.append(note.end - note.start)
        return notes, velocities, durations

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], [], []


def load_all_midis(folder):
    """Load all MIDI files from a folder and collect pitches, velocities and durations.

    This function expects `folder` to contain subfolders (e.g. composer names)
    and will recurse one level to find `.mid`/`.midi` files.
    """
    all_notes = []
    all_velocities = []
    all_durations = []

    if not os.path.isdir(folder):
        print(f"MIDI folder '{folder}' does not exist.")
        return all_notes, all_velocities, all_durations

    for composer in os.listdir(folder):
        composer_folder = os.path.join(folder, composer)
        if not os.path.isdir(composer_folder):
            continue
        for file in tqdm(os.listdir(composer_folder), desc=f"Scanning {composer}"):
            if file.lower().endswith(".mid") or file.lower().endswith(".midi"):
                file_path = os.path.join(composer_folder, file)
                notes, velocities, durations = midi_to_sequence(file_path)
                if len(notes) > 0:
                    all_notes.extend(notes)
                    all_velocities.extend(velocities)
                    all_durations.extend(durations)
    return all_notes, all_velocities, all_durations


def create_sequences(notes, velocities=None, durations=None, seq_length=20):
    """Split notes (and optionally velocities/durations) into input/output sequences.

    If velocities and durations are provided (lists of the same length as notes),
    this returns ((X_notes, X_vel, X_dur), (y_notes, y_vel, y_dur)).

    Otherwise it returns (X_notes, y_notes) as before.
    """
    inputs_notes, targets_notes = [], []
    inputs_vel, targets_vel = [], []
    inputs_dur, targets_dur = [], []

    has_extras = (velocities is not None and durations is not None)

    for i in range(len(notes) - seq_length):
        seq_in_notes = notes[i:i + seq_length]
        seq_out_note = notes[i + seq_length]

        inputs_notes.append(seq_in_notes)
        targets_notes.append(seq_out_note)

        if has_extras:
            seq_in_vel = velocities[i:i + seq_length]
            seq_out_vel = velocities[i + seq_length]
            seq_in_dur = durations[i:i + seq_length]
            seq_out_dur = durations[i + seq_length]

            inputs_vel.append(seq_in_vel)
            targets_vel.append(seq_out_vel)
            inputs_dur.append(seq_in_dur)
            targets_dur.append(seq_out_dur)

    X_notes = np.array(inputs_notes, dtype=np.int32)
    y_notes = np.array(targets_notes, dtype=np.int32)

    if not has_extras:
        return X_notes, y_notes

    X_vel = np.array(inputs_vel, dtype=np.int32)
    y_vel = np.array(targets_vel, dtype=np.int32)
    X_dur = np.array(inputs_dur, dtype=np.float32)
    y_dur = np.array(targets_dur, dtype=np.float32)

    return (X_notes, X_vel, X_dur), (y_notes, y_vel, y_dur)


def run_preprocessing(midi_folder, seq_length=20, save=True, test=False):
    """Run full preprocessing pipeline on a folder.

    Args:
        midi_folder (str): top-level folder containing subfolders with MIDI files.
        seq_length (int): sequence length for X inputs.
        save (bool): whether to save resulting .npy files.

    Returns:
        dict: contains raw lists and numpy arrays created (may be empty if not enough data).
    """
    print("Loading MIDI files...")
    all_notes, all_velocities, all_durations = load_all_midis(midi_folder)
    print(f"Total notes collected: {len(all_notes)}")

    results = {
        "all_notes": all_notes,
        "all_velocities": all_velocities,
        "all_durations": all_durations,
    }

    if len(all_notes) >= seq_length + 1:
        print("Creating sequences...")
        if all_velocities and all_durations and len(all_velocities) == len(all_notes) and len(all_durations) == len(all_notes):
            (X_notes, X_vel, X_dur), (y_notes, y_vel, y_dur) = create_sequences(all_notes, all_velocities, all_durations, seq_length=seq_length)
            results.update({
                "X_notes": X_notes,
                "y_notes": y_notes,
                "X_vel": X_vel,
                "y_vel": y_vel,
                "X_dur": X_dur,
                "y_dur": y_dur,
            })
            if save:
                if test:
                    np.save("Piano-Model/Testing Data/X_test.npy", X_notes)
                    np.save("Piano-Model/Testing Data/y_test.npy", y_notes)
                    np.save("Piano-Model/Testing Data/X_vel_test.npy", X_vel)
                    np.save("Piano-Model/Testing Data/y_vel_test.npy", y_vel)
                    np.save("Piano-Model/Testing Data/X_dur_test.npy", X_dur)
                    np.save("Piano-Model/Testing Data/y_dur_test.npy", y_dur)
                else:
                    np.save("Piano-Model/Training Data/X.npy", X_notes)
                    np.save("Piano-Model/Training Data/y.npy", y_notes)
                    np.save("Piano-Model/Training Data/X_vel.npy", X_vel)
                    np.save("Piano-Model/Training Data/y_vel.npy", y_vel)
                    np.save("Piano-Model/Training Data/X_dur.npy", X_dur)
                    np.save("Piano-Model/Training Data/y_dur.npy", y_dur)
        else:
            X_notes, y_notes = create_sequences(all_notes, seq_length=seq_length)
            results.update({"X_notes": X_notes, "y_notes": y_notes})
            if save:
                if test:
                    np.save("Piano-Model/Testing Data/X_test.npy", X_notes)
                    np.save("Piano-Model/Testing Data/y_test.npy", y_notes)
                else:
                    np.save("Piano-Model/Training Data/X.npy", X_notes)
                    np.save("Piano-Model/Training Data/y.npy", y_notes)
    else:
        print(f"Not enough notes to create sequences (need at least {seq_length + 1} notes).")

    # Always save raw velocities/durations if present
    if all_velocities and save:
        if test:
            np.save("Piano-Model/Testing Data/velocities_test.npy", np.array(all_velocities))
        else:
            np.save("Piano-Model/Training Data/velocities.npy", np.array(all_velocities))
    if all_durations and save:
        if test:
            np.save("Piano-Model/Testing Data/durations_test.npy", np.array(all_durations))
        else:
            np.save("Piano-Model/Training Data/durations.npy", np.array(all_durations))

    return results
