import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load preprocessed data
df = pd.read_csv('../ionosphere_preprocessing.csv')

# 2. Pisahkan fitur dan target
X = df[[f'Attribute{i}' for i in range(1, 35)]]
y = df['ClassNum']

# 3. Split data train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

clf = RandomForestClassifier(random_state=42)
gs = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

best_params = gs.best_params_
best_model = gs.best_estimator_

# 5. Manual logging ke MLflow
with mlflow.start_run(run_name="RandomForest_Tuning_Ionosphere"):
    # Log parameter hasil tuning
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    # Log metric training
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    mlflow.log_metric("train_accuracy", train_acc)
    # Log metric testing
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_acc)
    # Log classification report
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
    # Log model
    mlflow.sklearn.log_model(best_model, "model")
    print(f"Akurasi test: {test_acc:.4f}")
    print(f"Best params: {best_params}")
    print("Model dan artefak tuning sudah disimpan di MLflow Tracking UI.")

# Jalankan MLflow Tracking UI dengan: mlflow ui
