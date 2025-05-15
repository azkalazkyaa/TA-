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

# === STEP 2: Ekstraksi Fitur Statistik dari DWT Multilevel ===
def calculate_entropy(coeff):
    hist, _ = np.histogram(coeff, bins=100, density=True)
    hist = hist / np.sum(hist)
    return entropy(hist, base=2)

def extract_dwt_stat_features(audio_path, wavelet='db4', level=3):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.extend([
            np.mean(coeff),
            np.std(coeff),
            calculate_entropy(coeff),
            np.sum(coeff ** 2)  # energy
        ])
    return features

# === STEP 3: Labeling dan Ekstraksi Dataset ===
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
    create_feature_dataset("dataset_labels.csv")

def create_feature_dataset(csv_path, wavelet='db4', level=3):
    df = pd.read_csv(csv_path)
    all_features, audio_files, labels = [], [], []

    for _, row in df.iterrows():
        try:
            features = extract_dwt_stat_features(row['filename'], wavelet, level)
            all_features.append(features)
            audio_files.append(os.path.basename(row['filename']))
            labels.append(row['label'])
            print(f"Processed: {row['filename']}")
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    # Feature names
    stats = ['mean', 'std', 'entropy', 'energy']
    bands = [f"Level{i+1}" for i in range(level+1)]
    feature_names = [f"{band}_{stat}" for band in bands for stat in stats]

    feature_df = pd.DataFrame(all_features, columns=feature_names)
    feature_df['Audio File'] = audio_files
    feature_df['Label'] = labels
    feature_df = feature_df[['Audio File'] + feature_names + ['Label']]
    feature_df.to_csv("dwt_features_stat.csv", index=False)
    split_data("dwt_features_stat.csv")

# === STEP 4: Split Data ===
def split_data(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df.drop(['Audio File', 'Label'], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    train_df = pd.DataFrame(X_train)
    train_df['Label'] = y_train.values
    train_df['Audio File'] = df.loc[X_train.index, 'Audio File'].values

    test_df = pd.DataFrame(X_test)
    test_df['Label'] = y_test.values
    test_df['Audio File'] = df.loc[X_test.index, 'Audio File'].values

    train_df.to_csv("data_train.csv", index=False)
    test_df.to_csv("data_test.csv", index=False)

    print(f"Total data: {len(df)}")
    print(f"Training set: {len(train_df)} samples")
    print(f"Testing set: {len(test_df)} samples")

# === STEP 5: Training dan Evaluasi Random Forest ===
def train_and_evaluate():
    train_df = pd.read_csv("data_train.csv")
    test_df = pd.read_csv("data_test.csv")

    X_train = train_df.drop(['Audio File', 'Label'], axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop(['Audio File', 'Label'], axis=1)
    y_test = test_df['Label']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

    # Save model
    joblib.dump(model, "random_forest_model.pkl")
    print("Model disimpan sebagai 'random_forest_model.pkl'")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Stego", "Stego"], yticklabels=["Non-Stego", "Stego"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()

# === MAIN PIPELINE ===
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
