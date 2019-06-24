# CSC:Chinese Spelling Check


# SIGHAN 会议
## 对CSC任务进行评测，有标准数据集
## SIGHAN-7th 包含了2013年所有参加评测系统的实验结果
> [2013] Chinese Spelling Check Evaluation at SIGHAN Bake-off 2013
## 参评系统对应的论文列表：
https://dblp.uni-trier.de/db/conf/acl-sighan/acl-sighan2013.html

# Dataset
1. CLP 2014 Bake-off:http://ir.itc.ntnu.edu.tw/lre/clp14csc.html
2. SIGHAN-2013 Bake-off:http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html
------------------------------------------------------------------


# 以下三篇是复现难度不是很大的几种方法

## 基于CRF的汉语错误检测

> [SIGHAN-7 2013] Conditional Random Field-based Parser and Language Model for Traditional Chinese Spelling Checker
***Steps***
1. CRF分词，并找到被切分成连续字符的部分
2. 通过confusion set对怀疑错误的字符进行替换
3. LM判断替换是否合理

### CRF拓展
> [SIGHAN-8 ACL-2015] Word Vector/Conditional Random Field-based Chinese Spelling Error Detection for SIGHAN-2015 Evaluation
***Steps***
1. 分词
1. CRF判断词语是否是错误的
2. 通过confusion set对怀疑错误的字符进行替换
3. LM判断替换是否合理


## 基于N-gram的错误检测
> [SIGHAN-8 ACL-2015] Chinese Spelling Check System Based on N-gram Model
> 流程类似，但是引入了bi-gram和tri-gram两种LM，以及动态规划
> 详细介绍了confusing set怎么构造

--------------------------------------------------------------
# 以下论文的可行性不是很高，可供参考

## 基于LM+SMT
> [SIGHAN-7 ACL-2015] A Hybrid Chinese Spelling Correction Using Language Model and Statistical Machine Translation with Reranking
> 相较于传统的分词+LM方法，加入了SMT直接翻译错误句子，最后加入SVM进行reranking备选答案
> 实现难度较高

## 数据构造
> [EMNLP-18]A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Check
> 运用OCR和ASR的办法，将OCR识别错误的字符对作为训练数据
> 复现难度高

## 提出新的评估办法
> [AAAI-2018] A New Benchmark and Evaluation Schema for Chinese Typo Detection and Correction

## 只判断句子是否有错误，不修正
> [COLING-14] A Sentence Judgment System for Grammatical Error Detection
> rule-based + LM

## 基于粤语的错误检测
> [COLING-2016] ACE: Automatic Colloquialism, Typographical and Orthographic Errors Detection for Chinese Language
***Steps***
1. 分词，并找到被切分成连续字符的部分
2. 检测是否是粤语的表达
3. 通过confusion set对怀疑错误的字符进行替换
4. 语言模型LM判断替换是否合理


