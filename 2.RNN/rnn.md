## Syllabus

Mark notes related to:

* Language Models (LM): N-gram and Window-based Neural LMs
* Recurrent Neural Networks(RNN) and Gradient Vanishing/Exploding Problems
* Advanced RNN: LSTM/GRU
* Other RNNs

## Language Model (LM)
语言模型(Language Model)是对**语言序列概率分布的建模**，即其计算了当前序列出现的概率。给定一个单词序列，LM可以用来预测下一个词。

### n-gram
基于统计的方法。本质就是对出现的序列进行数数。

缺点：

* 稀疏问题严重
* 内存开销大

### Window-based Neural LMs

优缺点

## Recurrent Neural Networks(RNN)
RNN不是特指语言模型，只是一种用来对语言进行建模的方法，可以视为语言的encoder。

### 基本原理

### 评估标准

### Long-term Dependencies, Gradient Vanishing/Exploding

https://zhuanlan.zhihu.com/p/85776566

## Advanced RNN

### LSTM

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

为什么LSTM可以缓解RNN的梯度问题：
https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577

### GRU

### Bidirectional RNNs

### Multi-layer RNNs
