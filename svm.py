import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Fungsi klasifikasi dengan SVM
def classify_with_svm(train_file='data_train.csv', test_file='data_test.csv'):
    # Load dataset
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Pisahkan fitur dan label
    X_train = train_df.drop(['Audio File', 'Label'], axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop(['Audio File', 'Label'], axis=1)
    y_test = test_df['Label']

    # Buat model SVM dengan kernel RBF
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # Latih model
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    print("=== Hasil Klasifikasi SVM ===")
    print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Jalankan fungsi klasifikasi
classify_with_svm()
