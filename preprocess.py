import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from feature import extract_features
from data_split import create_splits, load_metadata, get_audio_path
from tqdm import tqdm



def process_file(args):
    row_idx, row, dataset_path = args

    audio_path = get_audio_path(dataset_path, row)

    try:

        features = extract_features(audio_path)
        return row_idx, features, row["classID"]

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return row_idx, None, None


def preprocess_split(df, dataset_path):

    tasks = [
        (idx, row, dataset_path)
        for idx, row in df.iterrows()
    ]

    X = []
    y = []

    max_workers = max(1, os.cpu_count() - 2)

    print(f"Extracting features using {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _, features, label in tqdm(
            executor.map(process_file, tasks),
            total=len(tasks)
        ):
            if features is not None:
                X.append(features)
                y.append(label)

    return X, np.array(y, dtype=np.int32)


def save_split(output_path, X, y):
    """
    Save variable-length features safely.
    """
    np.savez_compressed(
        output_path,
        X=np.array(X, dtype=object),
        y=y
    )


def main():
    DATASET_PATH = "UrbanSound8K"

    print("Loading metadata...")
    metadata = load_metadata(DATASET_PATH)

    print("Splitting metadata...")
    train_df, val_df, test_df = create_splits(metadata)

    print(
        f"Split sizes -> "
        f"Train: {len(train_df)}, "
        f"Val: {len(val_df)}, "
        f"Test: {len(test_df)}"
    )

    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Validation ----------------
    print("\n--- Processing validation split ---")

    X_val, y_val = preprocess_split(val_df, DATASET_PATH)

    print(
        f"Validation samples: {len(X_val)}, "
        f"Labels: {y_val.shape}"
    )

    save_split(
        os.path.join(output_dir, "val.npz"),
        X_val,
        y_val
    )

    # ---------------- Test ----------------
    print("\n--- Processing test split ---")

    X_test, y_test = preprocess_split(test_df, DATASET_PATH)

    print(
        f"Test samples: {len(X_test)}, "
        f"Labels: {y_test.shape}"
    )

    save_split(
        os.path.join(output_dir, "test.npz"),
        X_test,
        y_test
    )

    # ---------------- Train ----------------
    print("\n--- Processing train split ---")

    X_train, y_train = preprocess_split(train_df, DATASET_PATH)

    print(
        f"Train samples: {len(X_train)}, "
        f"Labels: {y_train.shape}"
    )

    save_split(
        os.path.join(output_dir, "train.npz"),
        X_train,
        y_train
    )

    # ---------------- Full Dataset ----------------
    print("\n--- Saving full dataset ---")

    X_all = X_train + X_val + X_test
    y_all = np.concatenate(
        [y_train, y_val, y_test],
        axis=0
    )

    save_split(
        os.path.join(output_dir, "all.npz"),
        X_all,
        y_all
    )

    print(
        f"Full dataset size: {len(X_all)}, "
        f"Labels: {y_all.shape}"
    )

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()