import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
# TRAIN
# ======================
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("Training done, acc:", acc)
