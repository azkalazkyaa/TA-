import os
import wave
import numpy as np
import pandas as pd
import pywt
import librosa
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# === STEP 1: Penyisipan Pesan ===
def hide_message(audio_path, message, output_path):
    audio = wave.open(audio_path, mode='rb')
    frames = audio.readframes(audio.getnframes())
    samples = list(frames)
    binary_message = ''.join(format(ord(c), '08b') for c in message) + '00000000'
    msg_index = 0

    for i in range(len(samples)):
        if msg_index < len(binary_message):
            samples[i] = (samples[i] & ~1) | int(binary_message[msg_index])
            msg_index += 1
        else:
            break

    output_audio = wave.open(output_path, mode='wb')
    output_audio.setparams(audio.getparams())
    output_audio.writeframes(bytes(samples))
    output_audio.close()
    audio.close()

# === Fitur Tambahan ===
def calculate_entropy(arr):
    hist, _ = np.histogram(arr, bins=100, density=True)
    hist = hist / np.sum(hist)
    return entropy(hist, base=2)

def lsb_entropy(audio_path):
    with wave.open(audio_path, 'rb') as f:
        frames = f.readframes(f.getnframes())
        samples = np.frombuffer(frames, dtype=np.uint8)
        lsb = samples & 1
        counts = np.bincount(lsb, minlength=2)
        probs = counts / np.sum(counts)
        return entropy(probs, base=2)

def extract_features(audio_path, wavelet='db4', level=3):
    y, sr = librosa.load(audio_path, sr=None, mono=True, duration=5.0)
    dwt_coeffs = pywt.wavedec(y, wavelet, level=level)
    dwt_stats = []
    for coeff in dwt_coeffs:
        dwt_stats.extend([
            np.mean(coeff),
            np.std(coeff),
            calculate_entropy(coeff),
            np.sum(coeff**2)
        ])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).flatten())
    rms = np.mean(librosa.feature.rms(y=y).flatten())
    lsb_ent = lsb_entropy(audio_path)
    return dwt_stats + [zcr, rms, lsb_ent]

# === Labeling ===
def labeling(input_folder, output_folder):
    dataset = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            dataset.append({"filename": os.path.join(input_folder, filename), "label": 0})
    for filename in os.listdir(output_folder):
        if filename.endswith(".wav"):
            dataset.append({"filename": os.path.join(output_folder, filename), "label": 1})

    df = pd.DataFrame(dataset)
    df.to_csv("dataset_labels.csv", index=False)
    extract_feature_dataset("dataset_labels.csv")

def extract_feature_dataset(csv_path, wavelet='db4', level=3):
    df = pd.read_csv(csv_path)
    features, files, labels = [], [], []
    for _, row in df.iterrows():
        try:
            f = extract_features(row['filename'], wavelet, level)
            features.append(f)
            files.append(os.path.basename(row['filename']))
            labels.append(row['label'])
            print(f"Extracted: {row['filename']}")
        except Exception as e:
            print(f"Error: {e} on {row['filename']}")
    base_stats = [f"Level{i+1}_{s}" for i in range(level+1) for s in ['mean','std','entropy','energy']]
    final_cols = base_stats + ['zcr', 'rms', 'lsb_entropy']
    df_feat = pd.DataFrame(features, columns=final_cols)
    df_feat['Audio File'] = files
    df_feat['Label'] = labels
    df_feat.to_csv("features_combined.csv", index=False)
    split_data("features_combined.csv")

def split_data(path):
    df = pd.read_csv(path)
    X = df.drop(['Audio File','Label'], axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    pd.concat([X_train, y_train, df.loc[X_train.index, ['Audio File']]], axis=1).to_csv("data_train.csv", index=False)
    pd.concat([X_test, y_test, df.loc[X_test.index, ['Audio File']]], axis=1).to_csv("data_test.csv", index=False)

# === Train & Evaluate ===
def train_and_evaluate():
    train_df = pd.read_csv("data_train.csv")
    test_df = pd.read_csv("data_test.csv")
    X_train = train_df.drop(['Audio File','Label'], axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop(['Audio File','Label'], axis=1)
    y_test = test_df['Label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    joblib.dump(model, "rf_model_enhanced.pkl")
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Stego','Stego'], yticklabels=['Non-Stego','Stego'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()

# === MAIN ===
input_folder = "audio/non-stego"
output_folder = "audio/stego"
os.makedirs(output_folder, exist_ok=True)
message = 'kami ini bonek mania kami selalu dukung persebaya dimana kau berada disitu kami karena kami bonek mania'
for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"stego_{i+1:03d}.wav")
        hide_message(input_path, message, output_path)
labeling(input_folder, output_folder)
train_and_evaluate()