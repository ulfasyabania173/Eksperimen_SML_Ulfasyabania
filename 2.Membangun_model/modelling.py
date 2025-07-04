import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load preprocessed data
df = pd.read_csv('../ionosphere_preprocessing.csv')

# 2. Pisahkan fitur dan target
X = df[[f'Attribute{i}' for i in range(1, 35)]]
y = df['ClassNum']

# 3. Split data train/test (pakai random_state agar konsisten)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# 5. Mulai experiment MLflow
with mlflow.start_run(run_name="RandomForest_Ionosphere"):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
    print(f"Akurasi test: {acc:.4f}")
    print("Model dan artefak sudah disimpan di MLflow Tracking UI.")

# Jalankan MLflow Tracking UI dengan: mlflow ui
