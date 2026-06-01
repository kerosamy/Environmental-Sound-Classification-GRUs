import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from gru_model import GRUClassifier

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 10

CLASS_NAMES = [
    "air_conditioner","car_horn","children_playing","dog_bark","drilling",
    "engine_idling","gun_shot","jackhammer","siren","street_music"
]

FEATURE_CONFIGS = {
    "mfcc": (0, 40),
    "mel": (40, 168),
    "rms": (168, 169),
    "all": (0, 169)
}


# =========================
# LOAD DATA
# =========================
import numpy as np

def load_data():
    X_train = np.load("processed_data/train.npz", allow_pickle=True)["X"]
    y_train = np.load("processed_data/train.npz", allow_pickle=True)["y"]

    X_val = np.load("processed_data/val.npz", allow_pickle=True)["X"]
    y_val = np.load("processed_data/val.npz", allow_pickle=True)["y"]

    X_test = np.load("processed_data/test.npz", allow_pickle=True)["X"]
    y_test = np.load("processed_data/test.npz", allow_pickle=True)["y"]

    # ---- FIX: convert to numeric tensor ----
    X_train = np.array(X_train.tolist(), dtype=np.float32)
    X_val   = np.array(X_val.tolist(), dtype=np.float32)
    X_test  = np.array(X_test.tolist(), dtype=np.float32)

    # ---- COMPUTE NORMALIZATION STATS (TRAIN ONLY) ----
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    # ---- APPLY NORMALIZATION ----
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================
# SPLIT FEATURES
# =========================
def split_features(X):
    out = {}
    for name, (a, b) in FEATURE_CONFIGS.items():
        out[name] = np.array([x[:, a:b] for x in X], dtype=np.float32)
    return out


# =========================
# TRAIN
# =========================
def train_model(model, X_train, y_train, X_val, y_val, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=4, factor=0.5),
        ModelCheckpoint(os.path.join(run_dir, "best.keras"), save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

def evaluate(model, X_test, y_test, name, run_dir, split_name="test"):
    pred = np.argmax(model.predict(X_test), axis=1)

    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average="macro")

    print(f"\n{name} [{split_name}]")
    print("Accuracy:", acc)
    print("F1:", f1)
    print(classification_report(y_test, pred, target_names=CLASS_NAMES))

    # ── Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(f"{name} [{split_name}]")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"cm_{split_name}.png"))
    plt.close()

    return acc, f1


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    train_sets = split_features(X_train)
    val_sets   = split_features(X_val)
    test_sets  = split_features(X_test)

    results = []

    for name in FEATURE_CONFIGS.keys():

        print("\n====================")
        print("Training:", name)
        print("====================")

        input_shape = train_sets[name].shape[1:]

        clf = GRUClassifier(
            input_shape=input_shape,
            num_classes=NUM_CLASSES
        )

        run_dir = os.path.join(RESULTS_DIR, name)

        train_model(
            clf.model,
            train_sets[name], y_train,
            val_sets[name],   y_val,
            run_dir
        )

        # ── Evaluate on both splits ───────────────────────────────
        val_acc,  val_f1  = evaluate(clf.model, val_sets[name],  y_val,  name, run_dir, split_name="val")
        test_acc, test_f1 = evaluate(clf.model, test_sets[name], y_test, name, run_dir, split_name="test")

        results.append({
            "model":    name,
            "val_acc":  val_acc,
            "val_f1":   val_f1,
            "test_acc": test_acc,
            "test_f1":  test_f1,
        })

    # ── Leaderboard ──────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv("results/leaderboard.csv", index=False)

    print("\nFINAL RESULTS")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()