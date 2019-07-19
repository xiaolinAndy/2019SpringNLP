# 2019SpringNLP

## Intro:

A research on Chinese Spelling Check System, the assignment project for NLP course in 2019 Spring

## Files:

+ `CSC.py`: The main code of CSC task containing data preprocessing, cadidate choosing and evaluation metrics calculating.
+ `LM_API.py`: The API for n-gram language model.
+ `word2vec_lm.py`: The code of language model based on word2vec.
+ `data/`: containing needed data.
+ `result/`: 
+ `LM_results/`: containing n-gram probs.
+ `reference/`: some paper that we refer to.
+ `svm-crf/`: containing codes of svm and crf for detecting mistake location  

## Usage:

Here is an example for running test results on sighan7 test data. More options can be used are noted in `CSC.py`

    python CSC.py --data_json data/sighan7_simple.json --data_seg_json data/sighan7_seg_simple.json --lm_choose 3-gram --cand_choose svm --res_svm data/sighan7_svm.json --save_file result/sighan7.txt

Some of the key options are:

+  lm_choose : `3-gram` for using n-gram model and `word2vec` for using LM based on word2vec
+  cand_choose : how to choose correction candidates, available options are: `consec, single, svm`
+  data_json : the preprocessed data ready to be corrected
+  data_seg_json : the segmented format of data_json
+  res_svm : the candidate chosen by svm 
+  save_file : the saving path of final result

Note: If you use word2vec as language model you need to download the related word2vec data from https://pan.baidu.com/s/1oM6XFPjZWoIYwO83F_uTnw (password: 7umy) and put it into `data/` folder.

## Requirments:

+ python3
+ jieba
+ bs4
+ snownlp
+ pickle
+ numpy
+ tensorflow
+ json
+ sklearn