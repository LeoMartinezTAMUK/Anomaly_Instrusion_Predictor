# Package: intrustionDetector
# File: feature_engineering.py

# This script is responsible for applying optimal feature selection

from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
import numpy as np


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     FeatureSelection()
To improve the accuracy and efficiency of our machine learning model, it is recommended to
apply a feature selection algorithm that can reduce the dimensionality of our dataset,
prevent overfitting, and accelerate training. The FeatureSelection() method can be employed
to select a specific feature selection algorithm and specify the number of features to be
chosen (k number) before initiating the training process.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def chi_squared_selection(X_train, X_test, y_train, k):
    chi2_scores, p_values = chi2(X_train, y_train)
    top_k_features = np.argsort(chi2_scores)[-k:]
    Xtrain_selected = X_train[:, top_k_features]
    Xtest_selected = X_test[:, top_k_features]
    return Xtrain_selected, Xtest_selected, top_k_features

def mutual_info_selection(X_train, X_test, y_train, k):
    mi_scores = mutual_info_classif(X_train, y_train)
    top_k_features = np.argsort(mi_scores)[-k:]
    Xtrain_selected = X_train[:, top_k_features]
    Xtest_selected = X_test[:, top_k_features]
    return Xtrain_selected, Xtest_selected, top_k_features

def L1_based_selection(X_train, X_test, y_train, k):
    clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=50)
    clf.fit(X_train, y_train)
    coefs = abs(clf.coef_[0])
    top_k_features = np.argsort(coefs)[-k:]
    Xtrain_selected = X_train[:, top_k_features]
    Xtest_selected = X_test[:, top_k_features]
    return Xtrain_selected, Xtest_selected, top_k_features


def feature_selection(X_train, X_test, algorithm, k, y_train):
    if algorithm == 'Chi-Squared':
        Xtrain_selected, Xtest_selected, top_k_features = chi_squared_selection(X_train, X_test, y_train, k)
        print(f"\nFeature Selection Utilized (K={k}): {algorithm}")
        isValid = True            
    elif algorithm == 'Mutual Info':
        Xtrain_selected, Xtest_selected, top_k_features = mutual_info_selection(X_train, X_test, y_train, k)
        print(f"\nFeature Selection Utilized (K={k}): {algorithm}")
        isValid = True           
    elif algorithm == 'L1-Based':
        Xtrain_selected, Xtest_selected, top_k_features = L1_based_selection(X_train, X_test, y_train, k)
        print(f"\nFeature Selection Utilized (K={k}): {algorithm}") 
        isValid = True   
    elif algorithm == 'None':
        Xtrain_selected = X_train
        Xtest_selected = X_test
        top_k_features = "default features"
        k = 115 # Default number of features after preprocessing
        print(f"\nFeature Selection Utilized (K={k}): {algorithm}")
        isValid = True
    else:
        Xtrain_selected = X_train
        Xtest_selected = X_test
        top_k_features = "default features"
        k = 115 # Default number of features after preprocessing
        isValid = False   
        print(f"\nThe specified algorithm '{algorithm}' is not available")
    return Xtrain_selected, Xtest_selected, top_k_features, isValid