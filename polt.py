import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

# ── constants ──
N_MFCC = 40
N_MELS = 128
SR = 22050
HOP_LENGTH = 512

# ── load dataset ──
data = np.load("processed_data/all.npz", allow_pickle=True)
X = data["X"]
y = data["y"]

# ── output folder ──
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

# ── plot 5 samples ──
for idx in range(5):

    features = np.array(X[idx], dtype=np.float32)

    print(f"\nSample {idx} | Label: {y[idx]} | Shape: {features.shape}")

    mfcc = features[:, :N_MFCC].astype(np.float32)
    mel_db = features[:, N_MFCC:N_MFCC + N_MELS].astype(np.float32)
    rms = features[:, N_MFCC + N_MELS:].squeeze().astype(np.float32)

    mfcc = np.nan_to_num(mfcc)
    mel_db = np.nan_to_num(mel_db)
    rms = np.nan_to_num(rms)

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=SR,
        hop_length=HOP_LENGTH
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # ===== MFCC =====
    img0 = librosa.display.specshow(
        mfcc.T,
        x_axis="time",
        sr=SR,
        hop_length=HOP_LENGTH,
        ax=axes[0]
    )
    axes[0].set_title("MFCC")
    fig.colorbar(img0, ax=axes[0])

    # ===== Mel Spectrogram =====
    img1 = librosa.display.specshow(
        mel_db.T,
        x_axis="time",
        y_axis="mel",
        sr=SR,
        hop_length=HOP_LENGTH,
        ax=axes[1]
    )
    axes[1].set_title("Mel Spectrogram")
    fig.colorbar(img1, ax=axes[1])

    # ===== RMS =====
    axes[2].plot(times, rms)
    axes[2].set_title("RMS Energy")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Energy")
    axes[2].grid(True)

    plt.tight_layout()

    # ── SAVE FIGURE ──
    save_path = os.path.join(save_dir, f"sample_{idx}_label_{y[idx]}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")