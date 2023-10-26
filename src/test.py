from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #clasify the columns:binary, multicategory and numbers
    binary_columns = []
    multicat_columns = []
    for tcolumn in working_train_df.select_dtypes(include='object').columns: # list categories from dataframe according to n.uniques
        if working_train_df[tcolumn].nunique() == 2:
            binary_columns.append(tcolumn)
        else:
            multicat_columns.append(tcolumn)

    # # Create ordinal encoder and fit 
    bin_encoder = OrdinalEncoder()
    bin_encoder.fit(working_train_df[binary_columns])

    working_train_df[binary_columns] = bin_encoder.transform(working_train_df[binary_columns]) # encode just the binary columns
    working_val_df[binary_columns] = bin_encoder.transform(working_val_df[binary_columns])
    working_test_df[binary_columns] = bin_encoder.transform(working_test_df[binary_columns])
    
    # # Create one hot encoder and fit 
    oh_encoder = OneHotEncoder()
    oh_encoder.fit(working_train_df[multicat_columns])
    
    def encode_with_onehot(df, multicat_columns):
        df_encoded = oh_encoder.transform(df[multicat_columns])
        df_encoded = pd.DataFrame(df_encoded.toarray(),
                                    columns = oh_encoder.get_feature_names_out(multicat_columns))
        df =  df.join(df_encoded)
        df.drop(columns=multicat_columns,inplace = True)
        return df
    
    working_train_df = encode_with_onehot(working_train_df.copy(),multicat_columns) # encode multicategory columns
    working_val_df = encode_with_onehot(working_val_df.copy(),multicat_columns)
    working_test_df = encode_with_onehot(working_test_df.copy(),multicat_columns)
    
    ## create imputer and fit 
    median_imputer = SimpleImputer(strategy='median')

    median_imputer.fit(working_train_df) # both, one hot encoder and imputer returns np array's
    
    working_train_df = pd.DataFrame(median_imputer.transform(working_train_df),columns = working_train_df.columns)
    working_val_df = pd.DataFrame(median_imputer.transform(working_val_df),columns = working_val_df.columns)
    working_test_df = pd.DataFrame(median_imputer.transform(working_test_df),columns= working_test_df.columns)
    
    # feature scaling 
    numerical_columns = working_train_df.select_dtypes(exclude = 'object').columns.tolist()
    mm_scaler = MinMaxScaler(feature_range=(0,1), clip = True)
    working_train_df[numerical_columns] = mm_scaler.fit_transform(working_train_df[numerical_columns])
    working_val_df[numerical_columns] = mm_scaler.fit_transform(working_val_df[numerical_columns])
    working_test_df[numerical_columns] = mm_scaler.fit_transform(working_test_df[numerical_columns])
    
    train_data = working_train_df.copy(deep=True).to_numpy()
    val_data = working_val_df.copy(deep=True).to_numpy()
    test_data = working_test_df.copy(deep=True).to_numpy()
    return train_data, val_data, test_data