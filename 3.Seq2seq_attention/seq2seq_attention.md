
## Syllabus

Mark notes related to:

* Neural Machine Translation (NMT) with Seq2Seq
* Attention Mechanism
* Evaluation of NMT: BLEU

## Neural Machine Translation with Seq2Seq

2014年，第一篇seq2seq论文问世；2016年，google翻译从统计方法换成seq2seq方法。为什么seq2seq这么神奇？什么是seq2seq？

Seq2seq(Sequence-to-sequence)模型由两个RNN的神经网络模型构成，也称为**encoder-decoder模型**。其输入和输出都是一个句子。Encoder负责把输入句子编码成一个固定大小的向量H，decoder通过学习把H解码成句子 (可以理解成语言模型，通过上文预测下一个词)。**Seq2seq可以用于机器翻译，对话系统，自动文摘，一般和Attention机制结合提高模型性能**。本文从机器翻译出发来解释这个模型。

不包含Attention机制的seq2seq(sequence-to-sequence)模型流程可以用下图来表示：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/seq2seq.png)

### Seq2Seq Architecture - Encoder

Encoder实质是个RNN网络，图中的hidden state可以来自vanilla/LSTM/GRU rnn，也可以把这些模型变得更复杂，变成bidirectional/multi-layer(stacked) rnn等等。实际训练中，encoder常用stacked LSTM。而且在机器翻译任务中，为了使得***输入句子***前面的词能对***输出句子***前面的词有更强的作用力，输入句子经常是倒序输入到encoder中。例如下图的stacked LSTM encoder：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/encoder.png)

还有biLSTM encoder:

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/bilstm.png)

相比于统计翻译模型（需要维护词语/词语或者短语/短语的概率分布），神经网络模型涉及的人力和工程量大大减少，而且可以实现端到端的翻译模型优化。有趣的是，谷歌翻译在他们的NMT模型输出设置多一个语言参数，指定语言类型，可以实现一对多的seq2seq参数共享。

### Seq2Seq Architecture - Decoder

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/decoder.png)

Decoder实质也是个RNN网络，但它和encoder不同的是，**它的输入不仅基于上一个hidden state，还取决于上一个output预测出来的单词(仅在test或predict阶段)。在train阶段时，decoder直接用目标句子作为输入**。

**然而在test或predict阶段**，需要通过hidden states的输出来挑选概率最大的输出词语序列(“概率乘积”转化为“log概率的加和”)，这个时候又会涉及效率慢的问题。穷尽所有可能的组合是不可能的，通常用beam search来解决这个问题，即每个timestep保留topk个概率加和最高的词(如下图), 最后选择概率和最高的作为输出。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/beamsearch.png)

Beam Search如果产生了<'end'>的标识符代表句子结束。最终我们会得到多个可能以<'start'>开始，以<end>结尾的句子，那么如何让decoder终止呢？一般是通过设置<'end'>或者timestep的个数来暂停beam search的过程。上图的概率公式存在一个问题，由于log小于0，随着句子的增长，概率会越来越小，这样导致短句往往能获得更高的分数。一般对分数再取长度的平均:

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/ave_beamsearch.png)

**在training阶段**，decoder每个timestep的输出和word2vec的处理方式一样，通过softmax后用交叉熵来最小化损失函数。但由于softmax存在计算量大的问题，又会引入类似hierachical softmax或negatve sampling的方法来解决这个问题。

## Attention Mechanism

Seq2seq有其局限性，所有不同长度的句子最终都会被编码成同个长度的向量；并且由于rnn的long-term dependency问题，**只用最后一个hidden state的输出作为decoder的输入**使得模型效果不理想。为此Attention机制(注意力机制)就应运而生了。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/attention.png)

那么attention机制是如何解决长序列的问题呢? 首先要了解它是如何运作的。

在attention的模型中，decoder当前的hidden state需要和encoder的每一个hidden state做点乘(最简单的方法)，然后通过softmax得到概率分布(soft alignment)。这个概率分布决定encoder各个hidden state在参与当前decoder计算的时候需要贡献的比重(weigted sum)。这样模型可以自己学习输入序列每个词应该占的比重(soft alignment)，即灵活地解决长序列的问题。

这个weighted sum可以和decoder当前的hidden state组合成更长的向量，并计算输出；也可以把它和decoder下一个timestep的input组合后进一步计算隐藏层，具体计算用公式表达如下：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/attention_cal.png)

Attention的计算方法有多种, 如下:

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/attention_diff_methods.png)

## Evaluation of MT: Bilingual Evaluation Understudy (BLEU)

BLEU基于n-gram，然后通过对比decoder的输出序列A的n-gram组合在训练集B（也称为reference）的n-gram组合的比例来计算翻译准确度。注意A的某个n-gram的最大计数不能超过B同个n-gram的数量。BLEU的公式如下：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/bleu_0.png)

其中：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/bleu_1.png)

由于在实验中发现，n越大，Pn这个比例会指数递减，于是每个n-gram的比率都要加一个指数衰减系数：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/bleu_2.png)

并且针对短句也做出惩罚。句子相比于reference越短，惩罚越大。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/bleu_3.png)


## More

* Bidirectional stacked LSTM (biLSTM)