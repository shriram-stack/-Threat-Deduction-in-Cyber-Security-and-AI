# app.py

from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from neural_network import build_neural_network, train_neural_network, evaluate_neural_network
from svm import build_svm_model, train_svm_model, evaluate_svm_model
from cnn import build_cnn_model, train_cnn_model, evaluate_cnn_model

app = Flask(__name__)

# Load the synthetic dataset
data = pd.read_csv('cybersecurity_dataset.csv')

# Separate features (X) and target variable (y)
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for CNN
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = rf_model.score(X_test_scaled, y_test)
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, y_pred_rf).ravel()
fpr_rf = (fp_rf / (fp_rf + tn_rf)) * 100 if (fp_rf + tn_rf) > 0 else 0

# Train and evaluate Neural Network
nn_model = build_neural_network(input_shape=(X_train_scaled.shape[1],))
nn_model = train_neural_network(nn_model, X_train_scaled, y_train)
accuracy_nn, _, fpr_nn = evaluate_neural_network(nn_model, X_test_scaled, y_test)

# Train and evaluate SVM
svm_model = build_svm_model()
svm_model = train_svm_model(svm_model, X_train_scaled, y_train)
accuracy_svm, _, fpr_svm = evaluate_svm_model(svm_model, X_test_scaled, y_test)

# Train and evaluate CNN
cnn_model = build_cnn_model(input_shape=(X_train_cnn.shape[1], 1))
cnn_model = train_cnn_model(cnn_model, X_train_cnn, y_train)
accuracy_cnn, _, fpr_cnn = evaluate_cnn_model(cnn_model, X_test_cnn, y_test)

# Prepare results for display
results = [
    {"AI Model": "Random Forest", "Accuracy (%)": f"{accuracy_rf * 100:.1f}",
     "False Positive Rate": f"{fpr_rf:.1f}%", "Application": "Malware Detection"},
    {"AI Model": "Neural Networks", "Accuracy (%)": f"{accuracy_nn * 100:.1f}",
     "False Positive Rate": f"{fpr_nn:.1f}%", "Application": "Threat Analysis"},
    {"AI Model": "SVM", "Accuracy (%)": f"{accuracy_svm * 100:.1f}",
     "False Positive Rate": f"{fpr_svm:.1f}%", "Application": "Intrusion Detection"},
    {"AI Model": "CNN", "Accuracy (%)": f"{accuracy_cnn * 100:.1f}",
     "False Positive Rate": f"{fpr_cnn:.1f}%", "Application": "Phishing Detection"},
]

@app.route('/')
def index():
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)