## Syllabus

Mark notes related to:

* Subword Models - Wordpieces
* Pre-trained Concepts (fine-tuning/network freezing) in LM: TagLM, Elmo, ULMfit
* Transformers: Attention is all you need
* Bert

## Subword Models

### Motivation

Subword model是一种介于word-level和character-level的方法。首先word-level模型不能处理**未登录词(OOV, Out-Of-Vocabulary)**，即训练集不存在但是验证集或者测试集存在的单词; 而char-level模型虽然能解决OOV问题，但它会使序列变长，在rnn模型中性能会受影响，最后输出从char到word的处理也耗费时间和影响正确性 （目前的pure char model似乎表现都不错）。


### Two Trends

而subword models结合了word-model和char-model的优势。目前有两大趋势：

#### Wordpieces

根据char在corpus出现的频率**对连续的字节进行两两结合**，这样就把corpus拆分成char或者连续的char序列。谷歌的bert模型就把输入都处理成wordpieces的格式。经典的有**BPE算法(Byte-Pair Encoding)**。
 
**BPE**只是一种字节编码方法。给定一个corpus，BPE能帮助产生目标词库。注意其关注的是**连续**的**两个**字节。举一个经典的例子。

1）假设我们最初的字典是 (前面对应词频)：

```
5 l o w
2 l o w e r
6 n e w e s t 
3 w i d e s t
```
那我们的词汇表初始化的时候必须先包含字符级别的：l, o, w, e, r, n, w, s, t, i, d。

2）然后我们观察到es出现了最高的9次，那我们把es结合，并且更新我们的词汇表为：l, o, w, e, r, n, w, s, t, i, d, es。

```
5 l o w
2 l o w e r
6 n e w es t 
3 w i d es t
```

3）紧接着我们观察到est出现了最高的9次，那我们把est结合，并且更新我们的词汇表为：l, o, w, e, r, n, w, s, t, i, d, es, est。

```
5 l o w
2 l o w e r
6 n e w est 
3 w i d est
```
以此类推，**最后设置一个wordpieces词汇表阈值或者直至下一个wordpiece频率为1即停止**。

#### Hybrid

只保持一个较小的常用word字典，其他的均用<'unk'>代替，并且如果输入输出的序列遇到<‘unk’>，即进入额外的rnn模型进行处理。如果出现两个rnn模型，loss需要相加。

### Other Inspiration from Subword Models

* FastText embeddings (FAIR of Fackbook)

* Represent words by char-embedding

## Pre-trained Models: TagLM, Elmo, ULMfit

### Context-specific Word Embeddings

迄今为止我们学习了训练word embedding的多种方法：word2vec, Glove和fasttext，它们提供了pre-trained word vectors (事先训练好的词向量)给大家直接用于nlp的不同任务训练。但是我们也发现了问题。首先，**词语是具有多义的，但pre-trained word vectors一个词只用一个向量表示**，我们想要的是更细粒度(fine-grained)的词向量；然后我们也发现，在语言模型(LM: predict next word)中，rnn在每个词的hidden state位置都能结合上下文产生这个词的encoding。那么，我们可以通过结合rnn学到**Context-specific Word Embeddings**吗？这样能提高下游nlp任务的表现吗？针对这个启发，有一系列包含pretraining思想的模型开始问世，并且在不同的nlp任务上取得突破。

### ELMo：Embeddings from Language Models

在2018年ELMo提出之前，这篇文章的第一作者在2017年发表了另一篇论文TagLM (Semi-supervised sequence tagging with bidirectional language models, cited by 245 on 2020/04/15)，其中心思想是用LM预训练的word embedding协助基于rnn模型的NER任务，大致的结构如下图：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/taglm.png)

其中word embeding和LM都是事先训练好的，然后直接放进左图的模型中进行NER任务的训练。ELMo (Deep contextualized word representations, cited by 2655 on 2020/04/15)也是基于这个思想进行进一步的改造。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/elmo.png)

ELMo用stacked biLSTM（文中为两层）在大数据集上进行LM的无监督训练。**最初上下文无关的词向量是通过cnn对字符级进行编码得到**。然后把forward和backward同一层每个词的hidden state拼接起来变成xi，用不同权重si来对xi加权求和，最后得到一个词的ELMo表示(之所以用不同层来表示，是作者在文章中指明不同层表示的是不同特征，比如底层表示语法，顶层表示语义)。这个向量随后会放进下游的nlp任务中参与训练，而si是通过训练来获取的。**ELMo通过这种方式成功地把word embedding变成动态，LM训练好之后，输入句子可以实时得到word embedding**。

ELMo预训练模型的加入提升了所有nlp下游任务的性能，超越了当时各个任务state-of-the-art的方法。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/elmo_exp.png)

### ULMfit: Universal Language Model Fine-tuning for Text Classification

2018年是让人惊喜的一年，除了ELMo，还有ULMfit也让人惊艳。ELMo是nlp预训练模型的鼻祖，而ULMfit开启了nlp迁移学习(transfer learning)和微调(fine-tuning)的新天地。

最早的迁移学习思想是来自于计算机视觉，我们可以在一个domain上训练好cnn模型后，迁移到另一个domain进行微调。第二个domain通常数据量少。而ULMfit也是借用了这种思想，先在通用大数据集wiki上对LM进行预训练(pretrain), 然后迁移到具体的nlp任务上，在目标域数据集上对LM进行微调。作者提出了discriminative fine-tuning(不同层用不同的学习率)，slanted triangular learning rates(学习率调整方法)。


在分类任务上取得很好的提升：
![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/ulmfit_exp.png)

## Transformers

在Transformer问世之前，[加入attention机制的seq2seq模型 (encoder-decoder)](https://github.com/fionattu/nlp_algorithms/blob/master/3.Seq2seq_attention/seq2seq_attention.md)在各个任务上已经取得了很大的提升，但rnn模型的顺序处理无法并行化，使得它的训练过程特别耗时；LSTM等门机制也只能缓解rnn的长依赖问题。rnn的优点是能考虑上下文信息对句子进行编码，既然attention这么强大且能自己学习上下文的权重，如果再加入词语顺序（order）的信息，是否也能取代rnn呢？2017年google发表了论文“Attention is all you need”，在机器翻译上取得了很好的效果 (英德翻译中，BLEU提升2）。他们在论文中提出transformer模型，后来一举成名的BERT也是基于transformer构建的。


**References:** 

* Transformer论文精读：Attention is all you need

* [用Jupyter一步步实现Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

* [外国博主Jay Alammar对前沿nlp模型ELMo/transformer/bert的详解](http://jalammar.github.io)

### Architecture

论文算法模型，依旧遵循encoder-decoder的结构。按照论文的设置，encoder和decoder均为6个transformer组成，最底层输入是word embedding (dim=512)，之后每一层的输入都为上一层的输出, dim均为512。如下图，左边为encoder，右边是decoder。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/transformer_whole.png)

在上图抽取一个encoder和decoder来看他们的内部组成，如下图所示：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/transformer.png)

在encoder里，输入(word embedding/new embedding)经过一个**self-attention模块**后，每个输入(new embedding)会进入一个**参数相同的全连接网络** - **FFNN (feed forward neural network)**。在decoder里，结构和上述的encoder相同，只不过在self-attention模块和FFNN之间，还加入了**encoder-decoder attention模块**，来学习不同输入的权重。

**Residual Connection**

观察到在上述的encoder/decoder里，每个模块都加入了residual connection，在rnn中已经学习到，这样的连接在较深的网络中可以减缓梯度消失或爆炸。

### Encoder/Decoder: Self-attention

接下来观察self-attention的内部结构。从下图来看，self-attention会对每个输入向量加入上下文的信息，然后输出一个新向量。其中x，z，r的个数分别等于输入的词语个数，并且x，z，r的维度保持不变。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/transformer_encoder.png)

**One-head Attention**

原文提出的Multi-head Attention是重复执行8个One-head Attention。首先我们来了解One-head Attention的流程。以下的例子表示第一个encoder的self-attention模块（第一个encoder的输入是word embedding，处理方式跟word2vec没有差别）：

(1) 首先每个输入**xi需要产生三个维度更小的vectors: query qi, key ki, value vi**(dim=64)。这三个vectors的产生方法是通过与三个矩阵进行点乘得出的，分别是WQ, WK, WV。**这三个矩阵是通过训练得到，并且对每个xi共享**。query qi, key ki, value vi正是attention机制的一个运用，只不过在seq2seq，这些向量是用hidden states，而transformer专门为每个输入产生了三个不同角色的vectors。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/self_attention_0.png)

(2) 得到qi, ki, vi后，我们就可以像在seq2seq中一样来计算attention score了。假设我们的sequence只有两个词：Thinking Machines，下图黑线条方框展示的是“Thinking”这个词的attention计算方法。我们假设我们的query是从“Thinking”发出的，所以我们用q1跟所有的keys(k1, k2)分别进行点乘(包含自己)，这样得出的score代表当我们对“Thinking”进行编码时，所有词语加在这个词上的权重。

(3) 接着论文对这个score进行细微的处理：除于keys的维度的平方根，可以使得梯度更平缓。最后所有分数经过softmax进行归一化得到概率。每个词的概率再和他们的vectors vi进行相乘得到weighted vectors，然后所有词语的weighted vectors加和得到“Thinking”的输出z1。这个z1经过处理后会输入transformer的ffnn模块。


![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/self_attention_1.png)

**注意在decoder中，attention只能和前序词语参与计算，因为从LM角度来讲，后面的词语对当前词语的预测没有影响。**

**Multi-head Attention**

文章中，作者说他们发现使用多个One-head Attention是“benefitial”的，于是他们提出Multi-head Attention的概念。Multi-head Attention具体是如何运作的呢？


我们可以想象每个head是不受影响的，我们可以同时做多次One-head Attention，但每次我们的线性变换矩阵WQ, WK, WV都是不一样的，所以根据同一个xi我们会得到不同的query qi, key ki, value vi，如下图：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_0.png)

这样如果我们有8个head，那最后我们得到8个不同的zi：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_1.png)

模型的做法不是对不同的zi取平均，而是把他们concat在一起，得到一个更大的zi，然后需要跟w0相乘:

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_2.png)

Multi-head Attention整个过程可以用下图表示，注意最终的z和x的维度一样(dim=512)。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_3.png)

Multi-head Attention确实能给模型带来性能的提升。通过查阅资料，发现有较多论文表明，不同层的attention是有其作用的，底层主要提取词语的语法特征，顶层主要提取语义特征。既然同一层的作用一样，attention的输入也一样，设置多头的作用是什么呢？目前有多种说法尚未得到充分证实：

1. 提取不同特征（有可视化发现的确每个头提取的pattern是不一样的）；
2. 等于ensemble功能
3. dropout功能；

论文还说明，head的数量不在多，2的表现差，但4，6，8的表现是差不多的。


### Encoder: Positional Encoding 

因为丢失掉RNN的word order信息，一开始word embedding还考虑了位置编码:

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_4.png)

### Decoder: Encoder-Decoder Attention

正如以上所提的，decoder和encoder一样，但是多加了一层Encoder-Decoder Attention，来选择encoder不同输出的权重。注意在这个模块中，key和value vectors是由最后一层encoder的输出计算出来的，而query vectors是由decoder每一层的self-attention的输出提供的：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/en_decoder_attention.png)

### 一些还要注意的细节：

**ffnn**：每个encoder/decoder的不同timestep共享参数，但不同的encoder/decoder参数独立。

**Layernorm的作用**: [batchnorm](https://arxiv.org/pdf/1502.03167.pdf)是最初提出来的对神经元输入进行规范化的方法, 后续提出的laynorm, groupnorm, instancenorm都是batchnorm的改良版本。

[Layernorm](https://zhuanlan.zhihu.com/p/54530247)是对一个样本的所有特征进行归一化。如下图，将残差X和attention网络输出Z相加后得到一个2x4的张量，然后对这8个值进行归一化。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/layernorm.png)

这里提一下ln和bn ([batchnorm](https://zhuanlan.zhihu.com/p/54171297))的不同。bn是沿着batch的方向，对同一个特征进行归一化。由于对同一个特征进行操作，bn比ln更好理解，bn在batch较大，即样本数较多时，取得的效果(loss)要优于ln；但是当样本数较小时，例如rnn的输入序列长短相差较多时，在后面的timestep样本已经比较少，导致bn失去全局统计的优势，效果要比ln差。

ln和bn的流程都是一样的 (下面用bn论文截图)。如下图：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/ln.png)

所有的方法都会对规范化后的数值进行调整(scale and shift)，最后加入两个可学习的参数对数值进行进一步处理。这是因为作者认为由于改变了特征的分布，可能损害数据本身的表达能力，例如x改变后可能导致激活函数的输入只在线性区域，失去激活函数本身的非线性转换功能 (是的，所有论文都假设规范化发生在激活函数之前):

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/scaled_shift_ln.png)


但实际发现，规范化在激活函数之后效果更好，所以当查看pytorch底层代码时，发现两个参数都有默认的取值，实则直接用了规范化后的数值:

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/pt_bn.png)



**multiplicative attention vs. additive attention**: 作者在文中指出，使用乘法(点乘)注意力比加法注意力在实践中更快，更节省空间(参数更少)。加法注意力实则为一个具有单隐层的神经网络，如下：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/attention_add_dot.jpeg)

文中指出，在d_k较小时(上图的h或s的维度)，两种机制表现类似；当d_k变大时，加法注意力的效果会超过点乘注意力的效果 (这篇论文有详细的对比实验：https://arxiv.org/abs/1703.03906)。

为什么d_k越大效果(最后收敛的loss)会越差呢？d_k越大，query和key需要点乘并加和的项也会变多(query和key每个维度的均值为0，方差为1)，最后得到的针对某个key的attention score均值为0，方差为d_k。如果有个别attention score较大 (数量级的差别)，经过softmax后，会把较大的概率分配给这个attention score对应的标签; 这就导致在反向传播中，这个softmax的梯度会趋近于0，导致梯度更新缓慢且困难 (参见[知乎](https://www.zhihu.com/question/339723385))。


于是文章对attention score进行放缩 (scaled dot product attention)，控制每个注意力分数的方差都为1，避免出现注意力分数过大导致softmax值一家独大的现象。有效地解决梯度更新困难的问题：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/scaled_dot_atte.png)

 
**multi-head的作用**: (感兴趣可以看看[这篇知乎](https://www.zhihu.com/question/341222779/answer/814111138)): 原文中作者指出使用多头是因为发现benefitial, 也就是实验发现多头效果更好，并没有理论支撑。从作者附上的appendix中，也可以看出同个单词的不同head, attend的单词也是不一样，于是我们可以理解，模型可以通过多头去自己学习不同的特征空间。

**soft attention (weighted attention score) vs. hard attention (max attention score)**: soft attention在于我们最后的输出是输入的注意力加权之和，也就是所有参与注意力计算的输入都会参与结果的计算；而hard attention只会选取注意力最高的输入。

**位置编码(Positional Encoding)的理解**: 由于注意力机制不能像rnn网络一样捕捉序列顺序，所以作者加入了位置编码信息来弥补这种缺陷。

* Adam optimizer的设置(加入warmup)
* Regularization: residual dropouts和label smoothing

* 模型结构深入理解：[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)



### GPT

### BERT

Bert的提出使得nlp 11种任务达到突破。基于深度的双向transformer原理，采用预训练+微调的思想 (pre-training and fine-tuning)。

**双向transformer**：

* 区分于GPT和ELMO的特点，好处在哪里呢？

**预训练两个子任务**：

* MLM (Masked Language Model)

* NSP (Next Sentence Prediction) 

**其他要注意的细节**：

* 输入embedding

* "[CLS]"和"[SEP]"

* Bert具备两种不同大小的模型

	* BERTBASE (L=12, H=768, A=12, Total Parameters=110M) 
	* BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M).

**下游任务的微调方法**：

## 论文精读

* TagLM: Semi-supervised sequence tagging with bidirectional language models
* ELMo: Deep contextualized word representations
* ULMfit: Universal Language Model Fine-tuning for Text Classification

## More
* performance of transformers
* tensor2tensor
* Residual Connection
* char-models
* google还采用sentencepieces方法, 参考[sentencepieces](https://github.com/google/sentencepiece)
* image/music transformer


