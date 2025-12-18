import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ======================
# DAGSHUB (ONLINE MLFLOW)
# ======================
dagshub.init(
    repo_owner="RhezaPriyaAnargya",
    repo_name="telco-churn-mlflow",
    mlflow=True
)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("namadataset_preprocessing/telco_preprocessed.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAIN & LOGGING
# ======================
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ---- Metrics ----
    mlflow.log_metric("accuracy", acc)

    # ---- Params ----
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.2)

    # ---- Confusion Matrix (Artifact 1) ----
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("training_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("training_confusion_matrix.png")

    # ---- Metric Info JSON (Artifact 2) ----
    metric_info = {
        "accuracy": acc,
        "model": "LogisticRegression",
        "dataset": "telco_preprocessed.csv"
    }

    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact("metric_info.json")

    # ---- Model ----
    mlflow.sklearn.log_model(model, "model")

    print("Training done, acc:", acc)
