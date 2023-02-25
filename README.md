## Overview

This is the code for the paper "Exploiting the Fairness-Utility Tradeoff for Mitigating Un-predefined Bias in Machine Classification"

By Zhoufei Tang, Tao Lu and Tianyi Li



## Demo

The  [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) is used in this demo code.

Run the code with the following command in python 3 with the package listed in the "requirements.txt":

```python
python MachineClassifer_BiasMitigation.py
```

the result is stored in the "Result/UCIAdult" folder. With a threshold $\epsilon=0.005$, the attributes’ distance to origin along iterations of bias mitigation is shown in the figure below.


![](https://github.com/zftang/MachineClassifer_BiasMitigation_beta/blob/main/Result/UCIAdult/distance2origin_0.005.png)
