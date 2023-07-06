# Package: intrustionDetector
# File: models.py

# This script is responsible for training each algorithm on the training set individually

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     train_model()
To train the machine learning model, this method requires the X and y training data from
our dataset as well as the name of the algorithm to be used. There are several algorithms
supported by this method, including KNN, Random Forest, SVM, XGBoost, Logistic Regression,
Naive Bayes, and Decision Tree. Once the training is complete, the method will return the
trained classifier which can be used for subsequent evaluation.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
#from imblearn.over_sampling import SMOTE


# Train the features of the dataset (x) relative to the label (y) using a supervised ML algorithm
# Weights were applied to handle imbalances in the training dataset for the class label

def train_model(X, y, algorithm, **kwargs): # Keyword arguments
    
    # Python dictionary containing every algorithm available for the program
    classifiers = {
        'Random Forest': RandomForestClassifier,
        'KNN': KNeighborsClassifier,
        'Logistic Regression': LogisticRegression,
        'Naive Bayes': GaussianNB,
        'Decision Tree': DecisionTreeClassifier,
        'SVM': SVC,
        'XGBoost': XGBClassifier
    }

    classifier_class = classifiers.get(algorithm)
    if classifier_class is None:
        print(f"Invalid algorithm name: {algorithm}")
        return None

    classifier_params = {
        
        # Default values set for hyperparameters if unspecified by the user
        'n_estimators': kwargs.get('n_estimators', 100),
        'random_state': 42,
        'class_weight': {0: 1, 1: 20} # Should be changed based on the dataset utilized
        
      # Specifically set hyperparameter values for optimal performance on the NSL-KDD dataset  
    } if algorithm == 'Random Forest' else {
        'n_neighbors': kwargs.get('n_neighbors', 10)
        
    } if algorithm == 'KNN' else {
        'random_state': 42,
        'C': kwargs.get('C', 5),
        'max_iter': kwargs.get('max_iter', 25000)
        
    } if algorithm == 'Logistic Regression' else {
        'var_smoothing': 0.0001
        
    } if algorithm == 'Naive Bayes' else {
        'criterion': 'entropy',
        'random_state': 42
        
    } if algorithm == 'Decision Tree' else {
        'kernel': 'rbf',
        'C': 500,
        'gamma': 0.1,
        'random_state': 42,
        'decision_function_shape': 'ovr',
        'class_weight': {0: 1, 1: 20}
        
    } if algorithm == 'SVM' else {
        'max_depth': 3,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42
        
    } if algorithm == 'XGBoost' else {}

    classifier = classifier_class(**classifier_params)
    
    try:
        classifier.fit(X, y)
        return classifier
    
    except ValueError:
        print("Training failed due to invalid algorithm input. Please check the values/name.\n")
        return None

