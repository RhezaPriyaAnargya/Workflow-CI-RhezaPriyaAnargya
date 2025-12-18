import pandas as pd
import mlflow
import mlflow.sklearn
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

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
# TRAIN (TANPA start_run)
# ======================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ======================
# LOGGING (PAKAI RUN OTOMATIS)
# ======================
mlflow.log_metric("accuracy", acc)

mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("max_iter", 1000)
mlflow.log_param("test_size", 0.2)

# ---- Artifact 1: Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.savefig("training_confusion_matrix.png")
plt.close()

mlflow.log_artifact("training_confusion_matrix.png")

# ---- Artifact 2: Metric Info JSON ----
metric_info = {
    "accuracy": acc,
    "model": "LogisticRegression",
    "dataset": "telco_preprocessed.csv"
}

with open("metric_info.json", "w") as f:
    json.dump(metric_info, f, indent=4)

mlflow.log_artifact("metric_info.json")

# ---- MODEL (INI YANG DICARI DOCKER) ----
mlflow.sklearn.log_model(model, "model")

# ======================
# SAVE RUN_ID UNTUK CI
# ======================
run_id = mlflow.active_run().info.run_id
with open("run_id.txt", "w") as f:
    f.write(run_id)

print("Training done")
print("Accuracy:", acc)
print("Run ID:", run_id)
