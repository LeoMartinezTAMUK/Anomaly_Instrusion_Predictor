# Package: intrustionDetector
# File: visualization.py

# This script is responsible for creating visualization of the models based on evaluation of their performance

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     visualize_model()
The function visualizes the performance of different models on a given test set.
It takes in three arguments, namely models, X_test, and y_test. Once the models are evaluated,
from the python dictionary 'models', the function displays a visualization of their performance
on the test set in the forms of a bar chart and several other confusion matrix heat maps.
    """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Visualize the model(s) used
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_model(models, X_test, y_test):

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
