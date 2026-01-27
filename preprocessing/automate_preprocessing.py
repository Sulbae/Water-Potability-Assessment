import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import logging

def preprocess_data() -> None:

    logging.basicConfig(level=logging.INFO)

    # Dataset Source
    data_folder = "../raw_dataset"
    file_name = "water_potability_raw.csv"

    data_path = os.path.join(data_folder, file_name)
    
    # Load dataset
    data = pd.read_csv(data_path)
    logging.info(f"Dataset berhasil di-load dari {data_path}")

    # Konfigurasi
    TEST_SIZE = 0.2

    # Classifier Data
    clf_train = data.iloc[:int(len(data) * (1 - TEST_SIZE))]
    clf_test = data.iloc[int(len(data) * (1 - TEST_SIZE)):]

    # Anomali Detection Data
    anom_data = data[data['Potability'] == 1]
    anom_X = anom_data.drop(columns=['Potability'])
    anom_X_train = anom_X.iloc[:int(len(anom_X) * (1 - TEST_SIZE))]
    anom_X_test = anom_X.iloc[int(len(anom_X) * (1 - TEST_SIZE)):]

    # Preprocessing Pipeline 
    def preprocessing_pipeline_schema(): 
        return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocess Classifier Data
    preprocess_clf = preprocessing_pipeline_schema()
    clf_train_X = clf_train.drop(columns=['Potability'])
    clf_train_y = clf_train['Potability']
    
    clf_X_preprocessed = preprocess_clf.fit_transform(clf_train_X)
    clf_X_preprocessed = pd.DataFrame(clf_X_preprocessed, columns=clf_train_X.columns)
    clf_df_scaled = pd.concat([clf_X_preprocessed, clf_train_y.reset_index(drop=True)], axis=1)
    ## save clf preprocessed data
    clf_df_scaled.to_csv('../preprocessing/preprocessing_clf.csv', index=False)

    # Preprocess Anomaly Detection Data
    preprocess_anom = preprocessing_pipeline_schema()
    anom_X_preprocessed = preprocess_anom.fit_transform(anom_X_train)
    anom_X_preprocessed = pd.DataFrame(anom_X_preprocessed, columns=anom_X_train.columns)
    anom_df_scaled = pd.concat([anom_X_preprocessed, anom_X_test], axis=1)
    ## save anom preprocessed data
    anom_df_scaled.to_csv('../preprocessing/preprocessing_anom.csv', index=False)
    
    # End
    logging.info("Preprocessing selesai dan data telah disimpan.")

if __name__ == "__main__":
    preprocess_data()