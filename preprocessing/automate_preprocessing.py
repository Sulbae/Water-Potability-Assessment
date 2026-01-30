import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path 
import logging

def preprocess_data() -> None:

    logging.basicConfig(level=logging.INFO)

    # Dataset Source
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / 'raw_dataset' / 'water_potability_raw.csv'
    
    # Load dataset
    data = pd.read_csv(data_path)
    logging.info(f"Dataset berhasil di-load dari {data_path}")

    # Konfigurasi
    TEST_SIZE = 0.2

    # Data Split
    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data['Potability'],
        random_state=42
    )

    # Classifier Data
    clf_X_train = train_df.drop(columns=['Potability'])
    clf_y_train = train_df['Potability']

    clf_X_test  = test_df.drop(columns=['Potability'])
    clf_y_test  = test_df['Potability']


    # Anomali Detection Data
    anom_train = train_df[train_df['Potability'] == 0].drop(columns=['Potability'])
    anom_test  = clf_X_test.copy()

    # Preprocessing Pipeline 
    def preprocessing_pipeline_schema(): 
        return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocess Classifier Data
    preprocess_clf = preprocessing_pipeline_schema()

    ## Train
    clf_X_train_preprocessed = preprocess_clf.fit_transform(clf_X_train)
    clf_X_train_preprocessed = pd.DataFrame(clf_X_train_preprocessed, columns=clf_X_train.columns)
    clf_train_preprocessed = pd.concat([clf_X_train_preprocessed, clf_y_train.reset_index(drop=True)], axis=1)
    ## Test  
    clf_X_test_preprocessed = preprocess_clf.transform(clf_X_test)
    clf_X_test_preprocessed = pd.DataFrame(clf_X_test_preprocessed, columns=clf_X_test.columns)
    clf_test_preprocessed = pd.concat([clf_X_test_preprocessed, clf_y_test.reset_index(drop=True)], axis=1)

    # Preprocess Anomaly Detection Data
    preprocess_anom = preprocessing_pipeline_schema()
    ## Train
    anom_train_preprocessed = preprocess_anom.fit_transform(anom_train)
    anom_train_preprocessed = pd.DataFrame(anom_train_preprocessed, columns=anom_train.columns)
    ## Test
    anom_test_preprocessed = preprocess_anom.transform(anom_test)
    anom_test_preprocessed = pd.DataFrame(anom_test_preprocessed, columns=anom_test.columns)

    # Export Preprocessed Data
    preprocessing_dir = BASE_DIR / 'preprocessing'
    os.makedirs(preprocessing_dir, exist_ok=True)
    ## save clf preprocessed data
    clf_train_preprocessed.to_csv(preprocessing_dir / 'clf_train_preprocessed.csv', index=False)
    clf_test_preprocessed.to_csv(preprocessing_dir / 'clf_test_preprocessed.csv', index=False)
    ## save anom preprocessed data
    anom_train_preprocessed.to_csv(preprocessing_dir / 'anom_train_preprocessed.csv', index=False)
    anom_test_preprocessed.to_csv(preprocessing_dir / 'anom_test_preprocessed.csv', index=False)

    # Export Preprocessing Pipelines
    artifacts_dir = BASE_DIR / 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    dump(preprocess_clf, artifacts_dir / 'preprocessing_pipeline_clf.pkl')
    dump(preprocess_anom, artifacts_dir / 'preprocessing_pipeline_anom.pkl')

    # End
    logging.info("Preprocessing selesai dan data telah disimpan.")

if __name__ == "__main__":
    preprocess_data()