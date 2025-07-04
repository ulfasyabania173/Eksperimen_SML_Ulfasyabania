import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import dagshub
import joblib

# Konfigurasi MLflow ke DagsHub

# Inisialisasi DagsHub dan MLflow
dagshub.init(repo_owner='ulfasyabania173', repo_name='Eksperimen_SML_Ulfasyabania', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ulfasyabania173/Eksperimen_SML_Ulfasyabania.mlflow")
mlflow.set_experiment("Ionosphere_RF_Tuning_DagsHub")
dagshub.auth.add_app_token("9403af7d2674e41fe6b4ac5406316bc9ad618ba7")

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

# 5. Manual logging ke MLflow DagsHub
with mlflow.start_run(run_name="RandomForest_Tuning_DagsHub"):
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
    # Log model sebagai artifact manual (kompatibel DagsHub)
    joblib.dump(best_model, "rf_model.joblib")
    mlflow.log_artifact("rf_model.joblib")
    # Log confusion matrix (tambahan)
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("confusion_matrix_TP", cm[1,1])
    mlflow.log_metric("confusion_matrix_TN", cm[0,0])
    mlflow.log_metric("confusion_matrix_FP", cm[0,1])
    mlflow.log_metric("confusion_matrix_FN", cm[1,0])
    # Log ROC AUC (tambahan)
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", auc)
    # Simpan model terbaik ke file
    joblib.dump(best_model, "rf_model.joblib")
    mlflow.log_artifact("rf_model.joblib")
    print(f"Akurasi test: {test_acc:.4f}")
    print(f"Best params: {best_params}")
    print("Model dan artefak tuning sudah disimpan di MLflow DagsHub.")

# Jalankan MLflow Tracking UI online di: https://dagshub.com/ulfasyabania173/Eksperimen_SML_Ulfasyabania.mlflow
