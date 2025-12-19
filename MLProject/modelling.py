import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    # Set experiment (aman)
    mlflow.set_experiment("workflow-ci-retail")

    # Load dataset
    df = pd.read_csv("retail_preprocessed/rfm_ready.csv")

    X = df[["MonetaryValue", "Frequency", "Recency"]]
    y = df["Cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Logging tambahan (AMAN)
    mlflow.log_metric("accuracy_manual", acc)
    mlflow.log_metric("f1_weighted_manual", f1)

    print("Accuracy:", acc)
    print("F1-score:", f1)


if __name__ == "__main__":
    main()
