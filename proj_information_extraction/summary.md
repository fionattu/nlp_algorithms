信息抽取可以帮助我们从半结构化和非结构化文本中提取有用信息，应用到下游任务中，最常见的是辅助知识图谱的构建。

本项目主要探究信息抽取的两大任务：

* **命名实体识别 (NER: Named Entity Recognition)**
* **关系抽取 (RE: Relation Extraction)**

# 早期方法：基于规则的抽取

## AC自动机

### KMP算法
### Trie树
### AC自动机

## 正则表达式


# 传统机器学习的方法

## HMM
## MEMM
## CRF

# 深度学习方法

## RNN-CRF
现在bilstm_crf为公认的表现最好的模型，由百度在2015年提出。
[Bidirectional LSTM-CRF Models for Sequence Tagging](https://github.com/fionattu/nlp_algorithms/blob/master/proj_information_extraction/papers/baidu15_bilstm_crf.pdf)。

主要难点：
1. 动态规划求所有路径分数(分母)
2. 维特比解码求出最优路径


提升: 基于预训练bert词向量的bilstm-crf模型

## CNN-CRF

# 近期前沿技术
需要了解的方法：

* ELMO 
* GPT 
* Bert

## Attention
## Transfer Learning
## Semi-supervised


