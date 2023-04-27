# Program was created in Spyder 5.2.2 Anaconda with Python 3.9
# Created by: Leo Martinez III in Spring 2023

# The program's primary function is to analyze labeled datasets of network traffic using supervised machine learning (ML) algorithms
# The program will decide if a network traffic instance is an anomaly or normal based on the NSL-KDD dataset.

# Import the required libraries
import pandas as pd
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# OPTIONAL
from sklearnex import patch_sklearn  
patch_sklearn() # Intel(R) Extension for Scikit-learn* to improve training computation performance
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

#%%-----------------------------------------------------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     LoadDataset()
To make use of a dataset, we need to develop a LoadDataset() function that can convert a
.csv file into a usable set of features. If the file is not found or the format is incorrect,
an error message will be shown.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the dataset
def LoadDataset(file_path): # Loads the network traffic dataset from a CSV file and returns it as a Pandas DataFrame.  
    try:
        dataset = pd.read_csv(file_path) # NSL-KDD Dataset (both training and testing dataset separately)
        return dataset
    
    except FileNotFoundError:
        print("\nFile not found. Please check the file path and try again.\n")
        return None
    except pd.errors.ParserError:
        print("\nUnable to load the file. Please check if the file is in the correct format.\n")
        return None
    
#%%-----------------------------------------------------------------------------------------------------------------------------

# Feature selection and data preprocessing before executing training/evaluation
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     FeatureSelection()
To improve the accuracy and efficiency of our machine learning model, it is recommended to
apply a feature selection algorithm that can reduce the dimensionality of our dataset,
prevent overfitting, and accelerate training. The FeatureSelection() method can be employed
to select a specific feature selection algorithm and specify the number of features to be
chosen (k number) before initiating the training process.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FeatureSelection(X_train, X_test, algorithm, k, y_train):
    try:
        if algorithm == 'Chi-Squared':
            
            # Feature Selection   
            chi2_scores, p_values = chi2(X_train, y_train) # Chi-Squared
            
            # Select top k features based on Chi-Squared scores
            top_k_features = np.argsort(chi2_scores)[-k:]
            
            # Extract top k features
            Xtrain_selected = X_train[:, top_k_features]
            Xtest_selected = X_test[:, top_k_features]
            
            print(f"\nFeature Selection Utilized (K={k}): {algorithm}")            
            return Xtrain_selected, Xtest_selected, top_k_features
        
        elif algorithm == 'Mutual Info':
            
            # Feature Selection   
            mi_scores = mutual_info_classif(X_train, y_train) # Mutual Information
            
            # Select top k features based on Mutual Information scores
            top_k_features = np.argsort(mi_scores)[-k:]
            
            # Extract top k features
            Xtrain_selected = X_train[:, top_k_features]
            Xtest_selected = X_test[:, top_k_features]
            
            print(f"\nFeature Selection Utilized (K={k}): {algorithm}")        
            return Xtrain_selected, Xtest_selected, top_k_features
        
        elif algorithm == 'L1-Based': # L1-Based Feature Selection
            
            # Feature Selection   
            clf = LogisticRegression(penalty='l1', solver='liblinear', random_state = 50) # L1-based feature selection
            clf.fit(X_train, y_train)
            
            # Select top k features based on L1-based feature selection
            coefs = abs(clf.coef_[0])
            top_k_features = np.argsort(coefs)[-k:]
            
            # Extract top k features
            Xtrain_selected = X_train[:, top_k_features]
            Xtest_selected = X_test[:, top_k_features]
            
            print(f"\nFeature Selection Utilized (K={k}): {algorithm}")   
            return Xtrain_selected, Xtest_selected, top_k_features
            
        elif algorithm == 'None': # Incomplete
            
            Xtrain_selected = X_train
            Xtest_selected = X_test
            top_k_features = "default features"
            k = 115 # Default number of features after preprocessing
            
            print(f"\nFeature Selection Utilized (K={k}): {algorithm}")
            return Xtrain_selected, Xtest_selected, top_k_features
        
    except ValueError:
        print("The input data is not valid. Please check if the data is in the correct format. \n")    

#%%-----------------------------------------------------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     PreprocessData()
To effectively use our Machine Learning algorithms, it is essential to prepare our data in
advance. This involves encoding categorical data, such as 'Protocol' into a binary sequence
and scaling features with high variance to enhance overall performance.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Preprocess the data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def PreprocessData(dataset):
    try:
        X = dataset.iloc[:, 1:-1].values # Do not include first and last feature (id and class label)
        y = dataset.iloc[:, -1].values # Only include last feature (class label) (Anomaly = 1, Normal = 0)

        # Encoding categorical data
        ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
        X = np.array(ct1.fit_transform(X))

        # Scaling numerical data
        ct2 = ColumnTransformer(transformers=[('scaler', MinMaxScaler(), [0, 4, 5, 9, 12, 15, 16, 21, 22, 30, 31])], remainder='passthrough')
        X = np.array(ct2.fit_transform(X))

        return X, y

    except ValueError:
        print("Preprocessing failed due to unexpected data type. Please check the dataset.\n")
        return None, None


#%%-----------------------------------------------------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     TrainModel()
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
from imblearn.over_sampling import SMOTE


# Train the features of the dataset (x) relative to the label (y) using a supervised ML algorithm
# Weights were applied to handle imbalances in the training dataset for the class label
def TrainModel(X, y, algorithm):
    if algorithm == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0:1, 1:20}) # Parameters of each Algorithm
    elif algorithm == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=10)
    elif algorithm == 'Logistic Regression':
        classifier = LogisticRegression(random_state=42, C=5, max_iter=25000)
    elif algorithm == 'Naive Bayes':
        smote = SMOTE() # Attempt to improve Naive Bayes performance specifically
        X_resampled, y_resampled = smote.fit_resample(X, y)
        classifier = GaussianNB(var_smoothing=0.0001)
        classifier.threshold = 0.3
        classifier.fit(X_resampled, y_resampled)
        return classifier 
    elif algorithm == 'Decision Tree':
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
    elif algorithm == 'SVM':
        classifier = SVC(kernel='rbf', C=500, gamma=0.1, random_state=42, decision_function_shape='ovr', class_weight={0:1, 1:20})
    elif algorithm == 'XGBoost':
        classifier = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)

    try:
        classifier.fit(X, y)
        return classifier

    except ValueError:
        print("Training failed due to invalid algorithm input. Please check the values/name.\n")
        return None

#%%-----------------------------------------------------------------------------------------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     EvaluateModel()
This approach builds on the TrainModel() method by taking the trained model classifier
as input and generating a set of evaluation metrics, including Accuracy, Confusion Matrix,
Precision, Recall, F1-Score, and AUC-Score. The Cross-validation score can be adjusted by
specifying the number of folds as an argument.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Evaluate a machine learning model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score, confusion_matrix

# Print a list of evaluation metrics for each algorithm utilized in training, use the return value of TrainModel() as parameter input
def EvaluateModel(classifier, X_test, y_test, algorithm):
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

    
#%%-----------------------------------------------------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     VisualizeModel()
The function visualizes the performance of different models on a given test set.
It takes in three arguments, namely models, X_test, and y_test. Once the models are evaluated,
from the python dictionary 'models', the function displays a visualization of their performance
on the test set in the forms of a bar chart and several other confusion matrix heat maps.
    """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Visualize the model(s) used 
import matplotlib.pyplot as plt
import seaborn as sns

def VisualizeModel(models, X_test, y_test):

    algorithms = []
    accuracy_scores = []
    error_scores = []

    for algorithm, classifier in models.items():
        if classifier is not None:
            try:
                algorithms.append(algorithm)
                y_pred = classifier.predict(X_test)
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                error_scores.append(np.std(y_test == y_pred) / np.sqrt(len(y_pred)))
            except AttributeError:
                        print("The input model is not valid. Please provide a valid classifier.\n")
            except ValueError:
                    print("The input data is not valid. Please check if the input data is in the correct format.\n")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# OPTIONAL (Used for Organization)
    """ Use this code snippet if you need the bar chart ordered by highest amount of accuracy
    #sorted_idx = np.argsort(accuracy_scores)[::-1]
    #algorithms = [algorithms[idx] for idx in sorted_idx]
    #accuracy_scores = [accuracy_scores[idx] for idx in sorted_idx]
    #error_scores = [error_scores[idx] for idx in sorted_idx] """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Visualize a bar chart to compare the individual algorithm performances
    fig, ax = plt.subplots()
    bars = ax.bar(algorithms, [score * 100 for score in accuracy_scores], width=0.7, yerr=error_scores, color=['#1F77B4', '#FF7F0E', '#2CA02C', '#991C1C', '#FFEC00', '#FFC0CB', '#A020F0'])
    ax.set_xlabel('Algorithm Utilized')
    ax.set_ylabel('Accuracy Score (%)')
    ax.set_title('Comparison of Algorithm Results')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.yaxis.grid(True)
    ax.set_ylim([0, 100]) # set y-axis fixed min/max

    # Add percentages on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.2f}%'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.show()
    
    # Visualize a confusion matrix heat map to highlight tp, fp, tn, and fn
    for algorithm, classifier in models.items():
        if classifier is not None:
            try:
                y_pred = classifier.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                ax = plt.subplot()
                sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                ax.set_title(f'Confusion Matrix for {algorithm}')
                ax.xaxis.set_ticklabels(['Normal', 'Anomaly'])
                ax.yaxis.set_ticklabels(['Normal', 'Anomaly'])
                plt.show()
                
            except AttributeError:
                    print("The input model is not valid. Please provide a valid classifier.\n")
            except ValueError:
                print("The input data is not valid. Please check if the input data is in the correct format.\n")

#%%-----------------------------------------------------------------------------------------------------------------------------

def main():
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """ PHASE I: PREPROCESSING
Our initial step is to preprocess each dataset, which involves encoding categorical data
and scaling numeric features with high variance to enhance the model's overall performance.
After the data preprocessing step, we apply feature selection techniques to both datasets
to further enhance the model's performance.
    """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Load the dataset
    train_dataset = LoadDataset('KDDTrain.csv')
    test_dataset = LoadDataset('KDDTest.csv')

    if (train_dataset is not None) & (test_dataset is not None):
        
        # Preprocess the datasets
        X_train, y_train = PreprocessData(train_dataset) # 67325 Normal classes (0) | 58623 Anomaly classes (1)
        X_test, y_test = PreprocessData(test_dataset) # 9711 Normal classes (0) | 12833 Anomaly classes (1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """ PHASE II: FEATURE SELECTION
Once the preprocessing stage is finished, we proceed to enhance our model's overall performance
by applying a feature selection algorithm that filters out irrelevant features from the datasets.
The FeatureSelection() method allows us to specify the desired algorithm and the number of
features (k) to select as arguments.
        """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Feature selection (X of the training set, X of the testing set, FS algorithm name, k number of features)
        Xtrain_selected, Xtest_selected, top_k_features = FeatureSelection(X_train, X_test, 'L1-Based', 45, y_train)
        
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
    algorithm. After training, the TrainModel() method returns the classifier results, which can
    be utilized as an argument in both the EvaluateModel() and VisualizeModel() methods.
        """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Train the models
        knn_classifier = TrainModel(Xtrain_selected, y_train, 'KNN')
        lr_classifier = TrainModel(Xtrain_selected, y_train, 'Logistic Regression')
        rf_classifier = TrainModel(Xtrain_selected, y_train, 'Random Forest')
        nb_classifier = TrainModel(Xtrain_selected, y_train, 'Naive Bayes')
        svm_classifier = TrainModel(Xtrain_selected, y_train, 'SVM')
        dt_classifier = TrainModel(Xtrain_selected, y_train, 'Decision Tree')
        xgb_classifier = TrainModel(Xtrain_selected, y_train, 'XGBoost')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """ PHASE IV: EVALUATION
Once we obtain the results from the previous method calls, we use those return values as
an argument to conduct performance evaluations for each algorithm. The EvaluateModel()
method returns a list of performance metrics, including Accuracy, Precision, Recall,
F1-Score, AUC, Cross-validation, and Confusion Matrix. To avoid bias, we utilize a
separate dataset specifically for testing purposes (KDDTest.csv) instead of using a
percentage split of the same dataset.
        """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
        # Evaluate the models
        EvaluateModel(knn_classifier, Xtest_selected, y_test, 'KNN')
        EvaluateModel(lr_classifier, Xtest_selected, y_test, 'Logistic Regression')
        EvaluateModel(rf_classifier, Xtest_selected, y_test, 'Random Forest')
        EvaluateModel(nb_classifier, Xtest_selected, y_test, 'Naive Bayes')
        EvaluateModel(svm_classifier, Xtest_selected, y_test, 'SVM')
        EvaluateModel(dt_classifier, Xtest_selected, y_test, 'Decision Tree')
        EvaluateModel(xgb_classifier, Xtest_selected, y_test, 'XGBoost')
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """ PHASE V: VISUALIZATION
After completing both the training and testing phases, we store all the returned classifiers
into a Python dictionary called 'models'. We use the 'models' dictionary as input for the
parameters of the VisualizeModel() method, which displays the performance comparisons of
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
        VisualizeModel(models, Xtest_selected, y_test)

    else:
        print("Missing one or more datasets.\n") # One or more of the datasets could not be found
        
#%%-----------------------------------------------------------------------------------------------------------------------------

# Execute the code        
main()
