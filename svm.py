# svm.py

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def build_svm_model():
    """
    Builds a Support Vector Machine (SVM) model.
    """
    model = SVC(kernel='linear', random_state=42)
    return model

def train_svm_model(model, X_train, y_train):
    """
    Trains the SVM model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_svm_model(model, X_test, y_test):
    """
    Evaluates the SVM model on the test set.
    Returns accuracy, confusion matrix, and False Positive Rate (FPR).
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # Calculate False Positive Rate (FPR)
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
    
    return accuracy, conf_matrix, fpr