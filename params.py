#---------------------------
# data information
# raw data path
DatasetName = 'UCIAdult'
DataPath = 'Data/UCIAdult/adult.data.csv'  


# data attribute names (the order must match the 'adult.data.csv' file)
data_attribute = [
    'age', 'workclass', 'fnlwgt', 'education', 'education num',
    'marital status', 'occupation', 'relationship', 'race', 'sex',
    'capital gain', 'capital loss', 'hours per week', 'native country',
    'income'
]

numerical_attribute = [
    'age', 'fnlwgt', 'education num', 'capital gain', 'capital loss',
    'hours per week'
]

categorical_attribute = [
    'workclass', 'education', 'marital status', 'occupation', 'relationship',
    'race', 'sex', 'native country', 'income'
]

label_Y = 'income'
label_O = 'sex'


#---------------------------
# max sub-distance order for the distance matrix, -1 represent full order
h_order_val = -1

#---------------------------
# distance to origin threshold value
epsilon_threshold_val_fix_bool = True
epsilon_threshold_val_fix = 0.005

#---------------------------
# If ture, the sub-distance is calculated by weighted average
weighted_bool = False

#---------------------------
# parallel n_jobs in joblib
num_jobs = 32