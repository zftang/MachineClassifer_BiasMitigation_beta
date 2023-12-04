import numpy as np
import pandas as pd
import random
from sklearn import metrics
from catboost import CatBoostClassifier

# Setting random seeds for reproducibility
seed_valid = 0
seed_catboost = 0

# Array to store different types of datasets (Raw, Removal, Mitigation)
datatype_arr = ['Raw', 'Removal', 'Mitigation']
result_summary_dict = {}

# Iterating through each type of dataset
for datatype in datatype_arr:
    print('-----------------------------------------------')
    print('Training Model for the data:', datatype)
    
    # Defining labels and categorical attributes
    label_Y = 'income'
    label_O = 'sex'
    categorical_attribute = [
        'workclass', 'education', 'marital status', 'occupation', 'relationship',
        'race', 'native country'
    ]
    # Setting CatBoost Classifier parameters
    cat_params = {
        'l2_leaf_reg': 0.7424788083358183,
        'learning_rate': 0.27533872636959533,
        'max_depth': 8,
        'iterations': 1000,
        'early_stopping_rounds': 20,
        'cat_features': categorical_attribute,
        'random_seed': seed_catboost
    }

    
    # Loading and preprocessing data depending on the dataset type
    if datatype == 'Raw':
        categorical_attribute_ = categorical_attribute+[label_O]

        df_data = pd.read_csv('Data/UCIAdult/Processed/data_train_'+datatype+'.csv', index_col=0)
        df_data[categorical_attribute_] = df_data[categorical_attribute_].astype('int')

        df_data_test = pd.read_csv('Data/UCIAdult/Processed/data_test_'+datatype+'.csv', index_col=0)
        df_data_test[categorical_attribute_] = df_data_test[categorical_attribute_].astype('int')

        cat_params['cat_features'] = categorical_attribute_

    else:
        df_data = pd.read_csv('Data/UCIAdult/Processed/data_train_'+datatype+'.csv', index_col=0)
        df_data[categorical_attribute] = df_data[categorical_attribute].astype('int')

        df_data_test = pd.read_csv('Data/UCIAdult/Processed/data_test_'+datatype+'.csv', index_col=0)
        df_data_test[categorical_attribute] = df_data_test[categorical_attribute].astype('int')

    # Initializing and training the CatBoost Classifier
    model = CatBoostClassifier(**cat_params)

    X = df_data.drop(columns=label_Y).copy()
    y = df_data[label_Y].copy()

    # Splitting the data into training and validation sets
    data_index = [i for i in range(len(y))]
    random.seed(seed_valid)
    random.shuffle(data_index)  

    train_len = np.int32(len(data_index) * 0.8)
    train_index = data_index[:train_len]
    valid_index = data_index[train_len:]


    X_train, X_valid = X.loc[train_index], X.loc[valid_index]
    y_train, y_valid = y.loc[train_index], y.loc[valid_index]

    
    # Fitting the model
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=10)

    
    
    #---------------------
    # Evaluate the performanace of the model on the test dataset
    # Calculate the Accuracy and Fairness metrics
    y_pred = model.predict(df_data_test.drop(columns=label_Y))
    y_true = df_data_test[label_Y].values

    df_data_test_ = pd.read_csv('Data/UCIAdult/Processed/data_test_Raw.csv', index_col=0)
    O_val = df_data_test_[label_O].values

    # Calculating accuracy and fairness metrics
    df_result = pd.DataFrame(index=range(len(y_true)))
    df_result['y_true'] = y_true
    df_result['y_pred'] = y_pred
    df_result[label_O] = O_val

    df_result_1 = df_result[df_result[label_O]==1]
    df_result_0 = df_result[df_result[label_O]==0]


    accuracy_val = metrics.accuracy_score(y_true, y_pred)
    statistical_parity  = np.abs(
    df_result_1[(df_result_1['y_pred'] == 1)].shape[0] / df_result_1.shape[0] - \
    df_result_0[(df_result_0['y_pred'] == 1)].shape[0] / df_result_0.shape[0]
    )


    equalized_odds = 0
    for y_val in [0, 1]:

        df_result_1 = df_result[(df_result[label_O]==1) & (df_result['y_true']==y_val)]
        df_result_0 = df_result[(df_result[label_O]==0) & (df_result['y_true']==y_val)]    

        equalized_odds  += np.abs(
        df_result_1[(df_result_1['y_pred'] == 1)].shape[0] / (df_result_1.shape[0] + 1e-8) - \
        df_result_0[(df_result_0['y_pred'] == 1)].shape[0] / (df_result_0.shape[0] + 1e-8)
        )   
    
    print('Result for data: ' + datatype)
    result_str = 'Accuracy=' + str(np.round(accuracy_val, 4)) + '; Delta_SP=' + str(np.round(statistical_parity, 4)) + '; Delta_EO=' + str(np.round(equalized_odds, 4))
    print(result_str)    
    
    
    result_summary_dict[datatype] = result_str
    
    
print('\n\n\n\n\n')
print('------------------------------------------------------')
print('Summary of the result of Bias Mitigation')

for datatype in datatype_arr:
    print(datatype+':\n', result_summary_dict[datatype])    