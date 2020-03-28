## Syllabus

Mark notes related to:

* Language Models (LM): N-gram and Window-based Neural LMs
* Recurrent Neural Networks(RNN) and Gradient Vanishing/Exploding Problems
* Advanced RNN: LSTM/GRU
* Other RNNs

## Language Model (LM)
语言模型(Language Model)是对**语言序列概率分布的建模**，即其计算了当前序列出现的概率。给定一个单词序列，LM可以用来预测下一个词。
我们使用输入法或者搜索引擎的补全推荐功能，就是LM的应用。LM的建模方法可以分为统计方法和神经网络方法。

### N-gram - 统计语言模型
首先要理解n-gram的意思。n-gram指的是有n个单词的序列，给定一个句子，n-gram确定了采样的方法。例如我们的句子是“the students opened their”, 我们想要预测第五个词，以下是根据不同的n分割出来的数据（n=1/2/3/4）。

```
unigrams: “the”, “students”, “opened”, ”their”
bigrams: “the students”, “students opened”, “opened their”
trigrams: “the students opened”, “students opened their”
4-grams: “the students opened their”
```

在n-gram中，第n个词出现的概率与前面n-1个词出现的概率有关。定义如下：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/ngram_1.png)

最后转换成计算n-gram和(n-1)-gram的联合概率，而联合概率本质在文本中对出现的序列进行统计，记录出现的次数，如下：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/ngram_2.png)


n-gram模型虽然简单，但也有一些比较严重的缺点：

* **n取值太小容易失去重要的上文信息**。如果上面的例子完整的句子是“as the proctor started the clock, the students opened their”，如果n=5，我们就会丢掉上半句话，导致得出book的概率降低。
* **稀疏问题严重**。当n增大的时候，sparsity problem变得严重。不是每个序列都出现在文本，这导致得出的n-gram概率分布很稀疏。一般利用smoothing方法赋予不曾出现的单词序列一个小的delta概率。
* **内存开销大**。需要把每个n-gram出现的次数存储在内存中便于后期计算，模型的大小随着n的增大而增大(一般n不会超过5)。

### Window-based Neural Language Model - 固定词窗神经网络语言模型
了解到n-gram的缺点，自然有人想到利用深度学习的模型来解决这个问题。一开始人们想出的神经网络模型和词向量的模型很像，其既可以学习词嵌入，又可以用来预测语言序列的下一个词。最后一层的输出使用softmax，然后通过最小化**softmax输出**和**第n个词的one-hot表示**的交叉熵来优化模型的参数。但这个模型也有缺点，增大采样序列的词语数量，权重矩阵W也会增大。而且单词词向量之间的权重没有共享，这样就学不到词语间的相互关系。

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/neural_LM.png)


## Recurrent Neural Networks(RNN)
RNN中文翻译为循环神经网络，结构参见下图。从图中我们可以总结出RNN的一些重要的特点：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/rnn.png)

* 模型的输入分先后顺序，后面的词依赖于前面的词；
* h指的是模型的隐藏层(hidden state，由多个神经元组成)，ht特指在t这个timestep(第t个输入)的时候隐藏层的值；
* 其中recurrent(循环)指的是计算ht的时候，作用于h(t-1)的权重矩阵Whh是相同的(hh维度)；
* 怎么用rnn的输入？输出可以只取最后一个隐藏层的输出(yt = sigmoid(ht))来预测下一个词的概率/其他任务；也可以综合考虑每个隐藏层的值(h1, h2，..., ht), 然后取max或者mean输入下一个神经网络层，用来解决情绪识别等任务；也可以往上再套一个rnn，用当前隐藏层的值作为上面rnn的输入；
* 注意rnn不是特指语言模型，只是一种用来对语言进行建模的方法，可以理解为自然语言序列的encoder。

### RNN LM的评估标准

计算rnn LM的loss，需要对每个timestep的输出计算loss (yt-1的输出是t个词的概率分布)，具体如下：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/rnn_loss.png)

评估模型好坏的标准：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/rnn_eval.png)

### Long-term Dependencies, Gradient Vanishing/Exploding

RNN虽然解决了Window-based Neural Language Model的缺点：可以处理任意长度的语言序列(因为权重共享，模型大小确定)，但其也存在一些严重的问题，导致长处无法发挥。

1. 输入有前后顺序，后面的单词的隐藏层计算需要依赖上一个词的隐藏层计算结果，导致计算无法并行，大大降低了训练速度；
* 存在严重的梯度消失和梯度爆炸问题，导致模型的亮点**long-term dependencies**无法学到。


**梯度消失和梯度爆炸是深度学习的普遍问题**，当网络越深时这个问题越明显。而rnn的权重矩阵Whh叠乘更加重这个问题。rnn为什么会出现梯度消失或爆炸？参考[rnn梯度消失和梯度爆炸推导](https://github.com/fionattu/nlp_algorithms/blob/master/pics/derivation/rnn_grad_vanish_explode.pdf)。


**出现梯度消失有什么后果呢？**在反向传播时，因为梯度消失，后面的梯度传到越前面的timesteps时会越小，这样后面的单词对于距离比较远的前面的单词的影响会很小，模型最终学不到两者的相互关系（Long-term Dependencies）。这样就失去了rnn的作用。

**而出现梯度爆炸又会有什么后果？**反向传播时参数更新的幅度太大，loss一直处于震荡状态，最差可能导致结果溢出(无穷，nan值)，无法达到全局最优解。解决方法：gradient clipping。当梯度达到一定的阈值，就把他们设置回一个小一些的数字。



## Advanced RNN

### LSTM

LSTM的结构

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/lstm.png)

其他资源：

* [如何理解LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [LSTM如何缓解梯度消失](https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577)


In LSTMs, however, the presence of the forget gate, along with the additive property of the cell state gradients, enables the network to update the parameter in such a way that the different sub gradients in 

### GRU

### Bidirectional RNNs

### Multi-layer RNNs
