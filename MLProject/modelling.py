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
with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.2)

    mlflow.sklearn.log_model(model, "model")

    print("Training done, acc:", acc)

    # SAVE RUN_ID FOR CI
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
