# machine-learning-models/README.md
"""
Machine Learning Models
=====================

A collection of machine learning models for various tasks.
"""

import os
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Split the dataset into features (X) and target (y)
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the saved model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions on the test data
predictions = loaded_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Model accuracy: {accuracy:.2f}')