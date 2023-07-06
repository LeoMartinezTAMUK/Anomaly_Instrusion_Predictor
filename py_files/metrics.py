# Package: intrustionDetector
# File: metrics.py

# This script is responsible for evaluating the performance of the ML algorithms and comparing results

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     evaluate_model()
This approach builds on the TrainModel() method by taking the trained model classifier
as input and generating a set of evaluation metrics, including Accuracy, Confusion Matrix,
Precision, Recall, F1-Score, and AUC-Score. The Cross-validation score can be adjusted by
specifying the number of folds as an argument.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Evaluate a machine learning model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)

# Print a list of evaluation metrics for each algorithm utilized in training, use the return value of TrainModel() as parameter input
def evaluate_model(classifier, X_test, y_test, algorithm):
    if classifier is None or X_test is None or y_test is None:
        print("Invalid input. Please check if the model, test data, and algorithm are provided.")
        return
    
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    confusion_mat = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr) * 100
        
    print(f"\nResults for {algorithm} algorithm:")
    print(f"Accuracy: {accuracy:.2f}%") # Accuracy
    print(f"Precision: {precision:.2f}%") # Proportion of predicted positives that are truly positive
    print(f"Recall: {recall:.2f}%") # Proportion of actual positives that are correctly identified
    print(f"F1-Score: {f1:.2f}%") # Harmonic mean of precision and recall
    print(f"AUC: {auc_score:.2f}%") # area under the ROC ((Receiver Operating Characteristic) curve
    print(f"Confusion matrix: {confusion_mat}") # Ratio of TP, FP, FN, & TN


    # Calculate cross-validation score (cv = 5 cross fold validation)
    cv_score = cross_val_score(classifier, X_test, y_test, cv=5, scoring='accuracy').mean() * 100
    print(f"Cross-validation score: {cv_score:.2f}%")
    return accuracy

