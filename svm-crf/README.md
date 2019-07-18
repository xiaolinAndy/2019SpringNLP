# CSC: 错误位置检测
## author:luyu
## 2019/7/17

1. 数据预处理

>`python preprocess.py`

*input: json file
param: 
TRAIN: True (training dataset) or False (test dataset)
return:
word_dict = {word:freq}
word_embed = {word:[embed]}
pos_dict = {pos:idx}
word_class = {word:class}*


2. SVM 模型训练

>`python SVM_model.py`

3. CRF 模型训练

>`python CRF_model.py`