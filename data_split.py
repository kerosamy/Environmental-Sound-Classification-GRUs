
import os
import pandas as pd

def load_metadata(dataset_path):
    csv_path = os.path.join(
        dataset_path,
        "metadata",
        "UrbanSound8K.csv"
    )

    return pd.read_csv(csv_path)


def get_audio_path(dataset_path, row):

    return os.path.join(
        dataset_path,
        "audio",
        f"fold{row['fold']}",
        row["slice_file_name"]
    )


def create_splits(metadata):

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

if __name__ == "__main__":
    DATASET_PATH = "UrbanSound8K"

    metadata = load_metadata(DATASET_PATH)

    train_df, val_df, test_df = create_splits(metadata)

    print("\nTrain Samples :", len(train_df))
    print("Validation Samples :", len(val_df))
    print("Test Samples :", len(test_df))

    classes = metadata["class"].unique()

