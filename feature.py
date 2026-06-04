import numpy as np
import librosa

def extract_features(audio_path, n_mfcc=40, n_mels=128, hop_length=512, sr=44100, fixed_frames=120):
    signal, sr = librosa.load(audio_path, sr=sr)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    rms = librosa.feature.rms(y=signal, hop_length=hop_length)

    min_frames = min(mfcc.shape[1], mel_db.shape[1], rms.shape[1])

    mfcc = mfcc[:, :min_frames]
    mel_db = mel_db[:, :min_frames]
    rms = rms[:, :min_frames]

    def fix_length(x, target):
        if x.shape[1] > target:
            return x[:, :target]  # trim
        elif x.shape[1] < target:
            pad = target - x.shape[1]
            return np.pad(x, ((0, 0), (0, pad)), mode='constant')
        return x

    mfcc = fix_length(mfcc, fixed_frames)
    mel_db = fix_length(mel_db, fixed_frames)
    rms = fix_length(rms, fixed_frames)

    features = np.vstack([mfcc, mel_db, rms]).T 
    return features