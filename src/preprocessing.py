from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    
    # Clasify the columns 
    numerical_columns = working_train_df.select_dtypes(exclude='object').columns.tolist()
    binary_columns = []
    multicat_columns = []
    for tcolumn in working_train_df.select_dtypes(include='object').columns: # list categories from dataframe according to n.uniques
        if working_train_df[tcolumn].nunique() == 2:
            binary_columns.append(tcolumn)
        else:
            multicat_columns.append(tcolumn)

    # transfom into arrays
    train_df = working_train_df[numerical_columns].to_numpy()
    val_df = working_val_df[numerical_columns].to_numpy()
    test_df = working_test_df[numerical_columns].to_numpy()

    # binary encoder
    bin_encoder = OrdinalEncoder()
    bin_encoder.fit(working_train_df[binary_columns])

    train_df = np.concatenate((train_df,bin_encoder.transform(working_train_df[binary_columns])),axis = 1)
    val_df = np.concatenate((val_df,bin_encoder.transform(working_val_df[binary_columns])),axis = 1)
    test_df = np.concatenate((test_df, bin_encoder.transform(working_test_df[binary_columns])),axis = 1)
    
    # onehot encoder
    oh_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    oh_encoder.fit(working_train_df[multicat_columns])

    train_df = np.concatenate((train_df,oh_encoder.transform(working_train_df[multicat_columns])),axis =1)
    val_df = np.concatenate((val_df,oh_encoder.transform(working_val_df[multicat_columns])),axis = 1)
    test_df = np.concatenate((test_df,oh_encoder.transform(working_test_df[multicat_columns])),axis = 1)

    median_imputer = SimpleImputer(strategy='median')

    median_imputer.fit(train_df) # both, one hot encoder and imputer returns np array's
    
    train_df = median_imputer.transform(train_df)
    val_df = median_imputer.transform(val_df)
    test_df = median_imputer.transform(test_df)

    # min max scaler
    mm_scaler = StandardScaler()
    mm_scaler.fit(train_df)
    train_df = mm_scaler.transform(train_df)
    val_df = mm_scaler.transform(val_df)
    test_df = mm_scaler.transform(test_df)
   
    return train_df, val_df, test_df