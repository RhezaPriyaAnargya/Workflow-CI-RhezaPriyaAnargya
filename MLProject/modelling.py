import pandas as pd
import mlflow
import mlflow.sklearn
import json
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

os.makedirs("output", exist_ok=True)

df = pd.read_csv("namadataset_preprocessing/telco_preprocessed.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model_type", "LogisticRegression")

    # Artifact 1
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.savefig("output/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("output/confusion_matrix.png")

    # Artifact 2
    with open("output/metric_info.json", "w") as f:
        json.dump({"accuracy": acc}, f)
    mlflow.log_artifact("output/metric_info.json")

    # 🔥 SIMPAN MODEL LOKAL
    mlflow.sklearn.save_model(model, "output/model")

print("Training finished")
