# ----------------- 🔥 Import Library 🔥 -----------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------- 🔥 Load Data 🔥 -----------------
train_df = pd.read_csv('data_train_filtered.csv')
test_df = pd.read_csv('data_test_filtered.csv')

# ----------------- 🔥 Pisahkan Fitur dan Label 🔥 -----------------
X_train_full = train_df.drop(['Audio File', 'Label'], axis=1)
y_train = train_df['Label']

X_test_full = test_df.drop(['Audio File', 'Label'], axis=1)
y_test = test_df['Label']

# ----------------- 🔥 Seleksi Fitur: Energy, Mean, Std 🔥 -----------------
selected_columns = [col for col in X_train_full.columns if 
                    'energy' in col.lower() or 
                    'mean' in col.lower() or 
                    'std' in col.lower()]

X_train = X_train_full[selected_columns]
X_test = X_test_full[selected_columns]

print(f"Total fitur terpilih: {len(selected_columns)}")
print("Fitur-fitur:", selected_columns)

# ----------------- 🔥 Training Model Logistic Regression 🔥 -----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------- 🔥 Prediksi dan Evaluasi 🔥 -----------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAkurasi: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ----------------- 🔥 Visualisasi Confusion Matrix 🔥 -----------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Non-Stego', 'Stego'], yticklabels=['Non-Stego', 'Stego'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
plt.tight_layout()
os.makedirs('models', exist_ok=True)
plt.savefig('models/confusion_matrix_logreg_selected.png')
plt.show()

# ----------------- 🔥 Simpan Model 🔥 -----------------
with open('models/logreg_energy_mean_std.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel Logistic Regression (Energy/Mean/Std) telah disimpan.")
