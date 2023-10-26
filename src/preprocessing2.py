from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

def another_preprocess(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)


    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    #clasify the columns:binary, multicategory and numbers
    binary_columns = []
    multicat_columns = []
    for tcolumn in working_train_df.select_dtypes(include='object').columns: # list categories from dataframe according to n.uniques
        if working_train_df[tcolumn].nunique() == 2:
            binary_columns.append(tcolumn)
        else:
            multicat_columns.append(tcolumn)
    
    # Create ordinal encoder and fit 
    bin_encoder = OrdinalEncoder()
    bin_encoder.fit(working_train_df[binary_columns])

    working_train_df[binary_columns] = bin_encoder.transform(working_train_df[binary_columns]) # encode just the binary columns
    working_val_df[binary_columns] = bin_encoder.transform(working_val_df[binary_columns])
    working_test_df[binary_columns] = bin_encoder.transform(working_test_df[binary_columns])
    
    # # Create one hot encoder and fit
    oh_encoder = OneHotEncoder(sparse_output= False, handle_unknown= 'ignore')
    oh_encoder.fit(working_train_df[multicat_columns])
    
    def encode_with_onehot(df, multicat_columns = multicat_columns):
        df_encoded = oh_encoder.transform(df[multicat_columns])
        df_encoded = pd.DataFrame(df_encoded,
                                    columns = oh_encoder.get_feature_names_out(multicat_columns))
        df =  df.join(df_encoded)
        df.drop(columns=multicat_columns,inplace = True)
        return df
    
    working_train_df = encode_with_onehot(working_train_df.copy(),multicat_columns) # encode multicategory columns
    working_val_df = encode_with_onehot(working_val_df.copy(),multicat_columns)
    working_test_df = encode_with_onehot(working_test_df.copy(),multicat_columns)
    
    #69 - 79
    # # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # # Again, take into account that:
    # #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    # #     working_test_df).
    # #   - In order to prevent overfitting and avoid Data Leakage you must use only
    # #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    # #     model to transform all the datasets.

    median_imputer = SimpleImputer(strategy='median')

    median_imputer.fit(working_train_df) # both, one hot encoder and imputer returns np array's
    
    working_train_df = median_imputer.transform(working_train_df)
    working_val_df = median_imputer.transform(working_val_df)
    working_test_df = median_imputer.transform(working_test_df)
    

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.
    mm_scaler = MinMaxScaler(feature_range=(0,1), clip = True)
    working_train_df = mm_scaler.fit_transform(working_train_df)
    working_val_df= mm_scaler.fit_transform(working_val_df)
    working_test_df = mm_scaler.fit_transform(working_test_df)

    return working_train_df, working_val_df, working_test_df