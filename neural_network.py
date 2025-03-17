# neural_network.py

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_neural_network(input_shape):
    """
    Builds a simple feedforward Neural Network model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),  # Input layer
        Dense(32, activation='relu'),  # Hidden layer
        Dense(1, activation='sigmoid')  # Output layer (binary classification)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(model, X_train, y_train):
    """
    Trains the Neural Network model.
    """
    history = model.fit(
        X_train, y_train,
        epochs=20, batch_size=32, validation_split=0.2, verbose=1
    )
    return model

def evaluate_neural_network(model, X_test, y_test):
    """
    Evaluates the Neural Network model on the test set.
    Returns accuracy, confusion matrix, and False Positive Rate (FPR).
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # Calculate False Positive Rate (FPR)
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
    
    return test_accuracy, conf_matrix, fpr