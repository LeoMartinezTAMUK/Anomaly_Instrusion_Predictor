# Package: intrustionDetector
# File: data_cleaning.py

# This script is responsible for loading the dataset(s) and preprocessing the data

# Import the required libraries
import pandas as pd
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     load_dataset()
To make use of a dataset, we need to develop a LoadDataset() function that can convert a
.csv file into a usable set of features. If the file is not found or the format is incorrect,
an error message will be shown.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_dataset(file_path):
    
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded CSV file as a DataFrame, or None if an error occurred.
    """
    
    if not isinstance(file_path, str) or not file_path.endswith('.csv'):
        print("Invalid file path. Please provide a path to a CSV file.")
        return None

    try:
        with open(file_path) as f:
            dataset = pd.read_csv(f)
        return dataset
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"Unable to load file {file_path}: {e}")
        return None


#%%-----------------------------------------------------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     preprocess_data()
To effectively use our Machine Learning algorithms, it is essential to prepare our data in
advance. This involves encoding categorical data, such as 'Protocol' into a binary sequence
and scaling features with high variance to enhance overall performance.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(dataset, num_cols_to_encode, num_cols_to_scale):

    try:      
        # Split data and labels
        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1].values

        # Encoding categorical data
        ct1 = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), num_cols_to_encode)], remainder='passthrough')
        X = np.array(ct1.fit_transform(X))

        # Scaling numerical data
        ct2 = ColumnTransformer(
            transformers=[('scaler', MinMaxScaler(), num_cols_to_scale)], remainder='passthrough')
        X = np.array(ct2.fit_transform(X))

        return X, y,

    except (ValueError, AttributeError) as e:
        raise Exception(f"Error in preprocessing data: {str(e)}")


