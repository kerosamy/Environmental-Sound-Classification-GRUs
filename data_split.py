
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio

def load_metadata(dataset_path):
    """
    Load UrbanSound8K metadata CSV.
    """
    csv_path = os.path.join(
        dataset_path,
        "metadata",
        "UrbanSound8K.csv"
    )

    return pd.read_csv(csv_path)


# ============================================================
# 2. Build Audio Path
# ============================================================

def get_audio_path(dataset_path, row):
    """
    Create full path of an audio file from metadata row.
    """

    return os.path.join(
        dataset_path,
        "audio",
        f"fold{row['fold']}",
        row["slice_file_name"]
    )


# ============================================================
# 3. Load Audio File
# ============================================================

def load_audio(audio_path, sr=22050):
    """
    Load WAV file.

    Parameters
    ----------
    audio_path : str
    sr         : target sample rate

    Returns
    -------
    signal : np.ndarray
    sr     : int
    """

    signal, sr = librosa.load(
        audio_path,
        sr=sr
    )

    return signal, sr


# ============================================================
# 4. Plot Waveform
# ============================================================

def plot_waveform(signal, sr):
    """
    Plot waveform.
    """

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(
        signal,
        sr=sr
    )

    plt.title("Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Plot Mel Spectrogram
# ============================================================

def plot_spectrogram(signal, sr):
    """
    Plot Mel Spectrogram.
    """

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    plt.figure(figsize=(12, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title("Mel Spectrogram")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Play Audio Inside Notebook
# ============================================================

def play_audio(signal, sr):
    """
    Play audio inside Jupyter Notebook.
    """

    return Audio(signal, rate=sr)


# ============================================================
# 7. Show Example From A Class
# ============================================================

def show_class_example(
    dataset_path,
    metadata,
    class_name
):
    """
    Show one example from a given class.
    """

    sample = metadata[
        metadata["class"] == class_name
    ].iloc[0]

    audio_path = get_audio_path(
        dataset_path,
        sample
    )

    signal, sr = load_audio(audio_path)

    print("=" * 50)
    print("Class :", class_name)
    print("File  :", sample["slice_file_name"])
    print("Fold  :", sample["fold"])
    print("=" * 50)

    plot_waveform(signal, sr)
    plot_spectrogram(signal, sr)

    return play_audio(signal, sr)


# ============================================================
# 8. Create Train / Validation / Test Splits
# ============================================================

def create_splits(metadata):
    """
    Assignment split:
        Train : folds 1-6
        Val   : folds 7-8
        Test  : folds 9-10
    """

    train_df = metadata[
        metadata["fold"].isin([1, 2, 3, 4, 5, 6])
    ]

    val_df = metadata[
        metadata["fold"].isin([7, 8])
    ]

    test_df = metadata[
        metadata["fold"].isin([9, 10])
    ]

    return train_df, val_df, test_df


# ============================================================
# 9. Print Dataset Statistics
# ============================================================

def dataset_statistics(metadata):
    """
    Print useful information about the dataset.
    """

    print("Total Samples :", len(metadata))
    print()

    print("Class Distribution:")
    print(metadata["class"].value_counts())

    print()
    print("Fold Distribution:")
    print(metadata["fold"].value_counts().sort_index())


# ============================================================
# ======================= EXAMPLE ============================
# ============================================================

if __name__ == "__main__":
    DATASET_PATH = "UrbanSound8K"

    # Load metadata
    metadata = load_metadata(DATASET_PATH)

    # Dataset statistics
    dataset_statistics(metadata)

    # Create splits
    train_df, val_df, test_df = create_splits(metadata)

    print("\nTrain Samples :", len(train_df))
    print("Validation Samples :", len(val_df))
    print("Test Samples :", len(test_df))

    # ------------------------------------------------------------
    # Display one example from each class
    # ------------------------------------------------------------

    classes = metadata["class"].unique()

