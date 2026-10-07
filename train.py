import pandas as pd
import subprocess
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Fetch DVC version hash
try:
    with open("data/train.csv.dvc", "r") as f:
        dvc_hash = [line.split()[-1] for line in f if "md5:" in line][0]
except Exception:
    dvc_hash = "manual-v1"

# 2. Load dataset
df = pd.read_csv("data/train.csv")
X = df[["feature1", "feature2"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. MLflow Experiment Setup
mlflow.set_experiment("DVC_MLflow_Integration")

n_estimators = 50
max_depth = 3

with mlflow.start_run():
    # Log DVC hash to connect the model with the exact data version
    mlflow.log_param("dataset_dvc_hash", dvc_hash)
    
    # Log hyperparameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and log metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    # Log model artifact
    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="random_forest_model",
    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
)
    
    print(f"Run completed! Accuracy: {acc:.2f} | DVC Hash: {dvc_hash}")