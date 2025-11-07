from preprocess import run_preprocessing
from model import PianoModel
import os


if __name__ == "__main__":
    # 1. Folder containing MIDI files (adjust if your data lives elsewhere)
    midi_folder = os.path.join("Piano-Model", "MIDI Datasets", "Classical", "Classical")
    # Run the full preprocessing pipeline and save outputs
    run_preprocessing(midi_folder, seq_length=20, save=True)
    
    # 2. You can now proceed to train your model using the preprocessed data.
    # data_splitter = DataSplitter()
    # (X_train, X_test, y_train, y_test), (X_train_vel, X_test_vel, y_train_vel, y_test_vel), (X_train_dur, X_test_dur, y_train_dur, y_test_dur) = data_splitter.split_data()

    model = PianoModel()
    # Train your model here using the split data
    # model.train(X_train, y_train, X_train_vel, y_train_vel, X_train_dur, y_train_dur)

    midi_folder_test = os.path.join("Piano-Model", "MIDI Datasets", "Classical", "Classical Piano")
    run_preprocessing(midi_folder_test, seq_length=20, save=True, test=True)