import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    clf_train, clf_test = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data['Potability'],
        random_state=42
    )

    # Anomali Detection Data
    anom_data = data[data['Potability'] == 1]
    anom_X = anom_data.drop(columns=['Potability'])
    anom_train, anom_test = train_test_split(
        anom_X,
        test_size=TEST_SIZE,
        random_state=42
    )

    # Preprocessing Pipeline 
    def preprocessing_pipeline_schema(): 
        return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocess Classifier Data
    preprocess_clf = preprocessing_pipeline_schema()

    ## Train
    clf_X_train = clf_train.drop(columns=['Potability'])
    clf_y_train = clf_train['Potability']
    
    clf_X_train_preprocessed = preprocess_clf.fit_transform(clf_X_train)
    clf_X_train_preprocessed = pd.DataFrame(clf_X_train_preprocessed, columns=clf_X_train.columns)
    clf_train_preprocessed = pd.concat([clf_X_train_preprocessed, clf_y_train.reset_index(drop=True)], axis=1)
    ## Test
    clf_X_test = clf_test.drop(columns=['Potability'])
    clf_y_test = clf_test['Potability']
    
    clf_X_test_preprocessed = preprocess_clf.transform(clf_X_test)
    clf_X_test_preprocessed = pd.DataFrame(clf_X_test_preprocessed, columns=clf_X_test.columns)
    clf_test_preprocessed = pd.concat([clf_X_test_preprocessed, clf_y_test.reset_index(drop=True)], axis=1)

    ## save clf preprocessed data
    clf_train_preprocessed.to_csv('../preprocessing/clf_train_preprocessed.csv', index=False)
    clf_test_preprocessed.to_csv('../preprocessing/clf_test.csv', index=False)

    # Preprocess Anomaly Detection Data
    preprocess_anom = preprocessing_pipeline_schema()
    ## Train
    anom_train_preprocessed = preprocess_anom.fit_transform(anom_train)
    anom_train_preprocessed = pd.DataFrame(anom_train_preprocessed, columns=anom_train.columns)
    ## Test
    anom_test_preprocessed = preprocess_anom.transform(anom_test)
    anom_test_preprocessed = pd.DataFrame(anom_test_preprocessed, columns=anom_test.columns)

    ## save anom preprocessed data
    anom_train_preprocessed.to_csv('../preprocessing/anom_train_preprocessed.csv', index=False)
    anom_test_preprocessed.to_csv('../preprocessing/anom_test.csv', index=False)
    
    # End
    dump(preprocess_clf, "../artifacts/preprocessing_pipeline_clf.pkl")
    dump(preprocess_anom, "../artifacts/preprocessing_pipeline_anom.pkl")
    
    logging.info("Preprocessing selesai dan data telah disimpan.")

if __name__ == "__main__":
    preprocess_data()