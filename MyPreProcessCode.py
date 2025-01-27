import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def missing_values(df):
    # Handleing columns with numeric values like (float and int)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    
    # Handles columns categorical values (object type) 
    # like strings, categories, or labels 
    # but can store integers or floats also
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
    
    return df
    # Function for Handleing missing values
def label_encoding(df):
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])
    return df

    # Function for normalizing the data
def normalize_min_max(df):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

    # Function for removing irrelevant information
def remove_irrelevant(df):
    df = df.drop(['id'], axis=1, errors='ignore')
    return df

    # Function that would call otheer functions to perform the preprocessing
def preprocess_data(data):
    #Replace non-standard missing values
    data = data.replace('Infinity', np.nan) 
    # the use of infinity represents the undefined or the extreme values
    # and they are replaced with NaN means not any number


    # Handeled missing values
    print("Handling missing values...")
    data = missing_values(data)
    
    # Removed irrelevant columns
    print("Removing irrelevant columns...")
    data = remove_irrelevant(data)
    
    # Performed label encoding
    print("Performing label encoding...")
    data = label_encoding(data)
    
    # Normalized the data
    print("Normalizing data...")
    data = normalize_min_max(data)
    
    return data

def main():
    file_path = r"C:\Users\radhika\OneDrive\Desktop\Zine projects\ZINE-ML-PROJECT_INITIAL\archive\UNSW_NB15_training-set.csv"
    print("Reading data...")
    data = pd.read_csv(file_path)     # Read the data
    
    # Preprocessd the read data
    processed_data = preprocess_data(data)
    
    # Saved the processed data
    output_path = r"C:\Users\radhika\OneDrive\Desktop\Zine projects\ZINE-ML-PROJECT_INITIAL\mytrainingdata.csv"
    print("Saving processed data...")
    processed_data.to_csv(output_path, index=False)
    print("Processing complete!")

if __name__ == "__main__":
    main()