# Package: intrustionDetector
# File: main.py (contains main() method) (use the local files in this folder to run the program)

# Program was created in Spyder 5.2.2 Anaconda with Python 3.9

# The program's primary function is to analyze labeled datasets of network traffic using supervised machine learning (ML) algorithms
# The program will decide if a network traffic instance is an anomaly or normal based on the NSL-KDD dataset.

# Files are imported from the local directory in the test version
from data_cleaning import load_dataset, preprocess_data
from feature_engineering import feature_selection
from models import train_model
from metrics import evaluate_model
from visualization import visualize_model

# Can both be modified in ROOT __init__.py or explicitly in the argument of preprocess_data
# Used for specifying which values need to be encoded/preprocessed before training
from __init__ import num_cols_to_encode, num_cols_to_scale

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# OPTIONAL
from sklearnex import patch_sklearn  
patch_sklearn() # Intel(R) Extension for Scikit-learn* to improve training computation performance
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


def main(): # main method containing methods from all other files contained in the project
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """ PHASE I: PREPROCESSING
Our initial step is to preprocess each dataset, which involves encoding categorical data
and scaling numeric features with high variance to enhance the model's overall performance.
After the data preprocessing step, we apply feature selection techniques to both datasets
to further enhance the model's performance.
    """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Load the dataset
    train_dataset = load_dataset('KDDTrain.csv') # 67325 Normal classes (0) | 58623 Anomaly classes (1)
    test_dataset = load_dataset('KDDTest.csv') # 9711 Normal classes (0) | 12833 Anomaly classes (1)

    if (train_dataset is not None) & (test_dataset is not None):
        
        # Preprocess the datasets
        X_train, y_train = preprocess_data(train_dataset, num_cols_to_encode, num_cols_to_scale)
        X_test, y_test = preprocess_data(test_dataset, num_cols_to_encode, num_cols_to_scale)

#%%


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """ PHASE II: FEATURE SELECTION
Once the preprocessing stage is finished, we proceed to enhance our model's overall performance
by applying a feature selection algorithm that filters out irrelevant features from the datasets.
The feature_selection() method allows us to specify the desired algorithm and the number of
features (k) to select as arguments.
        """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
        # Feature selection (X of the training set, X of the testing set, FS algorithm name, k number of features)
        Xtrain_selected, Xtest_selected, top_k_features, isValid = feature_selection(X_train, X_test, 'L1-Based', 45, y_train)
        #isValid aspect of the return value is still under development and will be used later on
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
# OPTIONAL (Used for Debugging)   
        # Check which features were chosen
        print("\nSelected features:", top_k_features)
            
        # Check the size of both datasets and ensure the feature number matches
        print("\nTraining data shape:", Xtrain_selected.shape)
        print("Test data shape:", Xtest_selected.shape) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """ PHASE III: TRAINING
    Next, we proceed to train each algorithm on the training set (KDDTrain.csv). The parameters
    required for training are the X and Y values of the training set along with the name of the
    algorithm. After training, the train_model() method returns the classifier results, which can
    be utilized as an argument in both the evaluate_model() and visualize_model() methods.
        """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Train the models
        knn_classifier = train_model(Xtrain_selected, y_train, 'KNN')
        lr_classifier = train_model(Xtrain_selected, y_train, 'Logistic Regression')
        rf_classifier = train_model(Xtrain_selected, y_train, 'Random Forest')
        nb_classifier = train_model(Xtrain_selected, y_train, 'Naive Bayes')
        svm_classifier = train_model(Xtrain_selected, y_train, 'SVM')
        dt_classifier = train_model(Xtrain_selected, y_train, 'Decision Tree')
        xgb_classifier = train_model(Xtrain_selected, y_train, 'XGBoost')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """ PHASE IV: EVALUATION
Once we obtain the results from the previous method calls, we use those return values as
an argument to conduct performance evaluations for each algorithm. The evaluate_model()
method returns a list of performance metrics, including Accuracy, Precision, Recall,
F1-Score, AUC, Cross-validation, and Confusion Matrix. To avoid bias, we utilize a
separate dataset specifically for testing purposes (KDDTest.csv) instead of using a
percentage split of the same dataset.
        """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
        # Evaluate the models
        evaluate_model(knn_classifier, Xtest_selected, y_test, 'KNN')
        evaluate_model(lr_classifier, Xtest_selected, y_test, 'Logistic Regression')
        evaluate_model(rf_classifier, Xtest_selected, y_test, 'Random Forest')
        evaluate_model(nb_classifier, Xtest_selected, y_test, 'Naive Bayes')
        evaluate_model(svm_classifier, Xtest_selected, y_test, 'SVM')
        evaluate_model(dt_classifier, Xtest_selected, y_test, 'Decision Tree')
        evaluate_model(xgb_classifier, Xtest_selected, y_test, 'XGBoost')
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """ PHASE V: VISUALIZATION
After completing both the training and testing phases, we store all the returned classifiers
into a Python dictionary called 'models'. We use the 'models' dictionary as input for the
parameters of the visualize_model() method, which displays the performance comparisons of
the algorithms through a bar chart and heat maps.
        """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
       # Visualize the results using comparison
        models = {
           'KNN': knn_classifier,
           'Log. R.': lr_classifier,
           'R. Forest': rf_classifier,
           'NB': nb_classifier,
           'SVM': svm_classifier, 
           'D. Tree': dt_classifier,
           'XGB': xgb_classifier}
    
        # Creat a bar chart and confusion matrix heat maps for performance comparison
        visualize_model(models, Xtest_selected, y_test)

    else:
        print("Missing one or more datasets.\n") # One or more of the datasets could not be found
 
        
#%%-----------------------------------------------------------------------------------------------------------------------------


# Execute the scripts        
main()


# Note: Naive Bayes only performs well with L1-Based feature selection
