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

The results for the UCIAdult dataset can be found in the 'Result/UCIAdult' folder. Specifically, the figure below shows the attributes' distance to origin along iterations of bias mitigation, with a threshold value $\epsilon=0.005$.

![](https://github.com/zftang/MachineClassifer_BiasMitigation_beta/blob/main/Result/UCIAdult/distance2origin_0.005.png)

The information regarding the bias mitigation steps is stored in the 'Result/UCIAdult/result_info_0.005.csv' file, where the suffix '0.005' denotes the threshold value of epsilon used for the analysis.
