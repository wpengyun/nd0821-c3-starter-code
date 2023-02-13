from starter.ml.model import train_model, compute_model_metrics, inference
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_train_model():
    X = np.random.rand(100, 10)
    y = np.random.randint(low=0, high=2, size=100)
    assert type(train_model(X, y)) == RandomForestClassifier


def test_compute_model_metrics():
    y = np.random.randint(low=0, high=2, size=100)
    preds = np.random.randint(low=0, high=2, size=100)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64


def test_inference():
    model = RandomForestClassifier()
    X = np.random.rand(100, 10)
    y = np.random.randint(low=0, high=2, size=100)
    model.fit(X, y)
    X_test = np.random.rand(100, 10)
    inference(model, X_test)
