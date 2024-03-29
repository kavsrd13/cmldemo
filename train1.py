# train.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
from mlflow.sklearn import log_model

# Start an MLflow run
with mlflow.start_run():
    
    # Load the Iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define and log parameters
    params = {
        "test_size": 0.3,
        "random_state": 42,
        "model_type": "LogisticRegression",
        "max_iter": 200,
    }
    mlflow.log_params(params)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=params["max_iter"])
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    mlflow.log_metrics(metrics)

    print(f"Logged metrics: {metrics}")
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the trained model and log as an artifact
    model_path = "iris_model.pkl"
    joblib.dump(model, model_path)
    logged_model = log_model(model, artifact_path="model")

    # Register the model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="IrisClassifier")

    print("Model and metrics are logged and model is registered in MLflow")
