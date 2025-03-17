# cnn.py

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

def build_cnn_model(input_shape):
    """
    Builds a Convolutional Neural Network (CNN) model.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),  # Convolutional layer
        MaxPooling1D(pool_size=2),  # Pooling layer
        Flatten(),  # Flatten layer
        Dense(50, activation='relu'),  # Fully connected layer
        Dense(1, activation='sigmoid')  # Output layer (binary classification)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(model, X_train, y_train):
    """
    Trains the CNN model.
    """
    history = model.fit(
        X_train, y_train,
        epochs=20, batch_size=32, validation_split=0.2, verbose=1
    )
    return model

def evaluate_cnn_model(model, X_test, y_test):
    """
    Evaluates the CNN model on the test set.
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