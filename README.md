## Overview

This is the code for the paper "Exploiting the Fairness-Utility Tradeoff for Mitigating Un-predefined Bias in Machine Classification"
By Zhoufei Tang, Tao Lu and Tianyi Li

Our code uses Python version 3.8.8 and relies on the packages listed in "requirements.txt". For reproducibility, we recommend using a virtual environment to manage your dependencies. If you have any questions about the code or the requirements, please feel free to open an issue or contact us directly.


## Demo

The  [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) is used in this demo code. The demo was run on a CPU AMD EPYC 7642, utilizing 32 parallel n_jobs in joblib. The execution time of the demo was 68 seconds.

Run the code with the following command in python 3 with the package listed in the "requirements.txt":

```python
python MachineClassifer_BiasMitigation.py
```

The results for the UCIAdult dataset can be found in the 'Result/UCIAdult' folder. Specifically, the figure below shows the attributes' distance to origin along iterations of bias mitigation, with a threshold value $\epsilon=0.005$. To reproduce the results or modify the parameters used, you can change the values in the params.py file. Please refer to the file for detailed explanations of each parameter.

![](https://github.com/zftang/MachineClassifer_BiasMitigation_beta/blob/main/Result/UCIAdult/distance2origin_0.005.png)

During each step, the attribute with the greatest distance to origin is iteratively modified until its distance to origin is less than the threshold value of $\epsilon$. In this demo, the information regarding the bias mitigation steps is stored in the 'Result/UCIAdult/result_info_0.005.csv' file, where the suffix '0.005' denotes the threshold value of $\epsilon$ used for the analysis.

```python
Step 0:
'Raw'
Step 1:
'relationship Iter1 re-bin: { Husband, Unmarried}'
'relationship Iter2 re-bin: { Unmarried, Not-in-family}'
'relationship Iter3 re-bin: { Not-in-family, Wife}' 
Step 2:
'marital status Iter4 re-bin: { Married-civ-spouse, Never-married}'
'marital status Iter5 re-bin: { Never-married, Divorced}'
Step 3:
'relationship Iter6 re-bin: { Wife, Own-child}'
Step 4:
'hours per week Iter7 polynomial: 3.0'
Step 5:
'marital status Iter8 re-bin: { Divorced, Widowed}'
Step 6:
'occupation Iter9 re-bin: { Craft-repair, Adm-clerical}'

```

After running the code [MachineClassifer_BiasMitigation.py](https://github.com/zftang/MachineClassifer_BiasMitigation_beta/blob/main/MachineClassifer_BiasMitigation.py), processed data files will be generated and stored in the path "Data\UCIAdult\Processed". These data are divided into training and test sets. Three types of data are included: raw data (Raw), data with the gender attribute removed (Removal), and transformed data using our bias mitigation technique (Mitigation). These data can be used to train a machine learning model, and the test set can be used to calculate evaluation metrics such as accuracy.

To train a machine learning model using the processed data files, 21 different classifiers were used in our paper. These classifiers are detailed on https://github.com/kathrinse/TabSurvey, where you can find instructions on how to use them. For simplicity, here we provide a demonstration of our method using the 'catboost' model. The hyperparameters are optimized with the validation set using the Python library 'Optuna'. To run the demonstration, please refer to the [Model_Catboost.py](https://github.com/zftang/MachineClassifer_BiasMitigation_beta/blob/main/Model_Catboost.py) file. Run the code with the following command in Python 3 with additional package 'catboost==1.0.6'.

```python
python Model_Catboost.py
```

On a CPU AMD EPYC 7642 utilizing 32 threads, the execution time was 32 seconds. The results of the demonstration show the effectiveness of our method in mitigating bias in the UCIAdult dataset. The results are reproducible using the random seed set in the   [Model_Catboost.py](https://github.com/zftang/MachineClassifer_BiasMitigation_beta/blob/main/Model_Catboost.py) file, and other seeds will produce similar results. 

```
------------------------------------------------------
Summary of the result of Bias Mitigation
Raw:
 Accuracy=0.8698; Delta_SP=0.1686; Delta_EO=0.1358
Removal:
 Accuracy=0.8698; Delta_SP=0.1507; Delta_EO=0.0854
Mitigation:
 Accuracy=0.8454; Delta_SP=0.0916; Delta_EO=0.0349
```

