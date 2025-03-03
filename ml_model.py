from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os

MODEL_FILE = "iris_model.pkl"

def train_and_save_model():
    """Train the model and save it"""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a RandomForest Classifier
    clf = RandomForestClassifier()

    # Train the model
    clf.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = clf.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save the trained model
    joblib.dump(clf, MODEL_FILE)
    print("Model saved as 'iris_model.pkl'")

# Train model if not already trained
if not os.path.exists(MODEL_FILE):
    train_and_save_model()

# Load trained model
def load_model():
    return joblib.load(MODEL_FILE)

