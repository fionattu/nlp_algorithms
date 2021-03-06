## Syllabus

Mark notes related to:

* Subword Models - Wordpieces
* Pre-trained Concepts (fine-tuning/network freezing) with LM: TagLM, Elmo, ULMfit
* Transformers: Attention is all you need
* Transformer-based Pretrained Models：GPT, Bert

## Subword Models

### Motivation

Subword model是一种介于word-level和character-level的方法。首先word-level模型不能处理**未登录词(OOV, Out-Of-Vocabulary)**，即训练集不存在但是验证集或者测试集存在的单词; 而char-level模型虽然能解决OOV问题，但它会使序列变长，在rnn模型中性能会受影响，最后输出从char到word的处理也耗费时间和影响正确性。


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

## Pre-trained Concepts: TagLM, Elmo, ULMfit

### Context-specific Word Embeddings

迄今为止我们学习了训练word embedding的多种方法：Word2Vec, GloVe和fastText，它们提供了pre-trained word vectors (事先训练好的词向量)给大家直接用于nlp的不同任务训练。但是我们也发现了问题。首先，**词语是具有多义的，但pre-trained word vectors一个词只用一个向量表示**，我们想要的是更细粒度(fine-grained)的词向量；然后我们也发现，在语言模型(LM: predict next word)中，rnn在每个词的hidden state位置都能结合上下文产生这个词的encoding。那么，我们可以通过结合rnn学到**Context-specific Word Embeddings**吗？这样能提高下游nlp任务的表现吗？针对这个启发，有一系列包含pretraining思想的模型开始问世，并且在不同的nlp任务上取得突破。

### ELMo：Embeddings from Language Models

在2018年ELMo提出之前，这篇文章的第一作者在2017年发表了另一篇论文TagLM (Semi-supervised sequence tagging with bidirectional language models, cited by 245 on 2020/04/15)，其中心思想是用LM预训练的word embedding协助基于rnn模型的NER任务，大致的结构如下图：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/taglm.png)

其中word embeding和LM都是事先训练好的，然后直接放进左图的模型中进行NER任务的训练。ELMo (Deep contextualized word representations, cited by 2655 on 2020/04/15)也是基于这个思想进行进一步的改造。TagLM直接用了ELMo LM最后一个hidden layer来丰富embedding表示，而ELMo是通过学习task-specific参数下
不同hidden layers的线性加合来最终取得输入的embedding。


![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/elmo.png)

ELMo用stacked biLSTM（文中为两层）在大数据集上进行LM的无监督训练。**最初输入的词向量是通过cnn对字符级进行编码得到**。然后把forward和backward同一层每个词的hidden state拼接起来变成xi，用不同权重si来对xi加权求和，最后得到一个词的ELMo表示(之所以用不同层来表示，是作者在文章中指明不同层表示的是不同特征，比如底层表示语法，顶层表示语义)。这个向量随后会放进下游的nlp任务中参与训练，而si是通过训练来获取的。**ELMo通过这种方式成功地把word embedding变成动态，LM训练好之后，输入句子可以实时得到word embedding**。

ELMo预训练模型的加入提升了所有nlp下游任务的性能，超越了当时各个任务state-of-the-art的方法。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/elmo_exp.png)

另外讨论一下一些实验发现。

下图为ELMo的表示方式，其中lamda是一个和具体下游任务有关的微调参数，如果设置比较大，假设为1，则最后得出的ELMo权重接近于每一层的平均表示；设置比较小，假设为0.001，模型则可以自己学到每一层embedding的占比权重。实验也证明使用0.001的效果较好 (在三个任务中f1提升了0.2)。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/elmo_obj.png)

文章还指出EMLo要怎么用？可以拼接word/char embedding作为下游模型的输入模块(放在inputs)，但作者也发现可以和rnn的hidden states进行拼接一起输出(放在outputs)，或者两者一起用，在某些任务上可以取得更好效果。在不同的任务中，不同的嵌入方式会发挥不同的作用，带来效果的提升。


### ULMfit: Universal Language Model Fine-tuning for Text Classification

2018年是让人惊喜的一年，除了ELMo，还有ULMfit也让人惊艳。ELMo是nlp预训练模型的鼻祖，而ULMfit开启了nlp迁移学习(transfer learning)和微调(fine-tuning)的新天地。

最早的迁移学习思想是来自于计算机视觉，我们可以在一个domain上训练好cnn模型后，迁移到另一个domain进行微调。第二个domain通常数据量少。而ULMfit也是借用了这种思想，先在通用大数据集wiki上对LM进行预训练(pretrain), 然后迁移到具体的nlp任务上，在目标域数据集上对LM进行微调。作者提出了discriminative fine-tuning(不同层用不同的学习率)，slanted triangular learning rates(学习率调整方法)。


在分类任务上取得很好的提升：
![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/ulmfit_exp.png)

## Transformers

在Transformer问世之前，[加入attention机制的seq2seq模型 (encoder-decoder)](https://github.com/fionattu/nlp_algorithms/blob/master/3.Seq2seq_attention/readme.md)在各个任务上已经取得了很大的提升，但rnn模型的顺序处理无法并行化，使得它的训练过程特别耗时；LSTM等门机制也只能缓解rnn的长依赖问题。rnn的优点是能考虑上下文信息对句子进行编码，既然attention这么强大且能自己学习上下文的权重，如果再加入词语顺序（order）的信息，是否也能取代rnn呢？2017年google发表了论文“Attention is all you need”，在机器翻译上取得了很好的效果 (英德翻译中，BLEU提升2）。他们在论文中提出transformer模型，后来一举成名的BERT也是基于transformer构建的。


**References:** 

* Transformer论文精读：Attention is all you need

* [第一作者用Jupyter一步步实现Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

* [10分钟带你深入理解Transformer原理及实现](https://zhuanlan.zhihu.com/p/80986272) 

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

(2) 得到qi, ki, vi后，我们就可以像在seq2seq中一样来计算attention score了。假设我们的sequence只有两个词：Thinking Machines，下图黑线条方框展示的是“Thinking”这个词的attention计算方法。我们假设我们的query是从“Thinking”发出的，所以我们用q1跟所有的keys(k1, k2)分别进行点乘(包含自己)，这样**得出的score代表当我们对“Thinking”进行编码时，所有词语加在这个词上的权重**。

(3) 接着论文对这个score进行细微的处理：除于keys的维度的平方根，可以使得梯度更平缓。最后所有分数经过softmax进行归一化得到概率。每个词的概率再和他们的vectors vi进行相乘得到weighted vectors，然后所有词语的weighted vectors加和得到“Thinking”的输出z1。这个z1经过处理后会输入transformer的ffnn模块。


![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/self_attention_1.png)

**注意在decoder中，attention只能和前序词语参与计算，因为从LM角度来讲，后面的词语对当前词语的预测没有影响。**

**Multi-head Attention**

文章中，作者说他们发现使用多个One-head Attention是“benefitial”的，于是他们提出Multi-head Attention的概念。Multi-head Attention具体是如何运作的呢？


我们可以想象每个head是不受影响的，我们可以同时做多次One-head Attention，**但每次我们的线性变换矩阵WQ, WK, WV都是不一样的**，所以根据同一个xi我们会得到不同的query qi, key ki, value vi，如下图：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_0.png)

这样如果我们有8个head，那最后我们得到8个不同的zi：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_1.png)

模型的做法不是对不同的zi取平均，而是把他们concat在一起，得到一个更大的zi，然后需要跟w0相乘:

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_2.png)

Multi-head Attention整个过程可以用下图表示，注意最终的z和x的维度一样(dim=512)。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_3.png)

**multi-head的作用**: (感兴趣可以看看[这篇知乎](https://zhuanlan.zhihu.com/p/69919890))。原文中作者指出使用多头是因为发现benefitial, 也就是实验发现多头效果更好，并没有理论支撑。从作者附上的appendix中，也可以看出同个单词的不同head, attend的单词也不一样。于是我们可以这样理解：模型的多头attention参数(K,Q,V矩阵)不共享，可训练参数多，赋予了模型专注于不同位置的能力，也就是说模型可以通过多头去自己学习不同特征空间的注意力机制，比如不同的子空间可能关注指代，依存等不同句法关系，最后综合表示成各个子空间的关联关系，这样极大提升了attention的表现能力。


### Encoder: Positional Encoding 

位置编码的维度和word embedding的维度相同：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/multihead_4.png)

**位置编码(Positional Encoding)的理解**: 由于注意力机制不能像rnn网络一样捕捉序列顺序，所以作者加入了位置编码信息来弥补这种缺陷：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/positional_encoding.png)

其中，i是位置向量的index, pos是词语在句子中的index。文中指出，这种编码方式有利于体现不同词语的相对位置。如[参考资料](https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model)指出，我们比较不同词语的同个i，其实是三角函数的平移(相位差)，也就是文中指出的，pos+k的encoding可以表示成pos的线性组合：例如sin(pos+k) = sin(pos)cosk + cos(pos)sink。

在实验结果中，作者指出把固定的正弦位置编码改成让模型自己学习的位置编码，发现实验结果和base模型差不多。

### Decoder: Encoder-Decoder Attention

正如以上所提的，decoder和encoder一样，但是多加了一层Encoder-Decoder Attention，来选择encoder不同输出的权重。注意在这个模块中，key和value vectors是由最后一层encoder的输出计算出来的，而query vectors是由decoder每一层的self-attention的输出提供的：

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/en_decoder_attention.png)

### 一些还要注意的细节：

**ffnn**：transformer的ffnn做了两次线性变换(输入输出dim=512)，相当一个具备单隐层的全连接网络 (激活函数为ReLU, hidden_dim=2048)。注意在同个encoder/decoder layer中，**ffnn对于每个输入是独立的并共享参数，但不同的encoder/decoder layer参数互相独立**。

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


**soft attention (weighted attention score) vs. hard attention (max attention score)**: soft attention在于我们最后的输出是输入的注意力加权之和，也就是所有参与注意力计算的输入都会参与结果的计算；而hard attention只会选取注意力最高的输入。

* Adam optimizer的设置(加入warmup)
* Regularization: residual dropouts和label smoothing

## Transformer-based Pretrained Models

### GPT1.0

首个利用transformer进行特征提取的预训练+微调模型。属于单向的自回归模型，利用12层的transformer decoder(带mask), 使得self-attention只关注上文信息。并且作者提出在四种nlp任务的适配结构：分类，文本蕴含，文本相似性，多选问题，都是基于transformer+linear的结构。

实验用BPE编码处理输入，使用**包含长句的数据集**对LM进行预训练并取得较低的perplexity，**使LM具备捕捉长依赖的能力**(ELMo会对标点符号进行断句，故只能处理短句)。并且作者还发现, 在优化不同nlp下游任务时**同时对LM进行微调**，可以**提升模型的泛化能力，也能加速收敛**。

实验发现9/12的数据集达到sota (比ensemble好)，并且发现迁移的LM层数越多，效果越好，进而证明预训练模型的作用。


### BERT (Bidirectional Encoder Representations from Transformers)

Bert紧接着GPT1.0提出，文中对现有的预训练模型进行分类：

* **Feature-based Approach**: 类似ELMo这样的，在较大的corpus上训练LM(没有labeled data，被视为无监督学习) ，固定LM参数成为预训练模型。下游任务的输入可以先通过LM得到固定的embeddings(上下文相关)，再输入具体的任务(不会对embeddings进行调整)。
* **Fine-tuning Approach**: 在大的corpus预训练LM后，根据下游任务调整叠加的模型参数(linear，bilstm_crf等)，并同时微调LM的参数。

Bert的提出基于Fine-tuning Approach，并在实验中使得11种nlp任务达到突破，其结构是堆叠**transformer的encoder层**(不使用decoder层，从全名也可以看出)，达到抽取语言深层特征的目的，最后连接FFNN+softmax完成特定任务。Bert采用预训练+微调的思想 (pre-training and fine-tuning)，模型提供预训练参数，微调是留给使用者进行的任务。

* Bert具备两种不同大小的模型 (其中L/H/A分别为num_encoder_layers/hidden_size/num_self_attention_heads)

	* BERT_BASE (L=12, H=768, A=12, Total Parameters=110M) 
	* BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M).

**双向transformer**：

* 区分于GPT和ELMO的特点，好处在哪里呢？

**预训练两个子任务**：

* **数据处理：使用"[CLS]"和"[SEP]"**。Bert使用**"[CLS]"**标识符 (classification token)作为每个训练样本的开始字符，这个字符对应位置的最后一层的hidden state，将作为输入的表示，服务于**文本分类任务**; 使用"[SEP]"标识符来分割sentence (预训练的NSP任务，微调的QA, NLI任务等)。

* **预训练任务1：MLM (Masked Language Model)**。文中指出单向LM (GPT使用的unidirectional decoder), 或者直接联合正反rnn的hidden states (elmo使用的biLM)的方法，都不能很好地体现双向的优势。于是作者提出了MLM的方法，既可以同时训练双向的LM，又可以防止一个词看见它自己(see itself)。"自己看见自己"意思是在**多层双向**encoder中，每个词的输入包括了它自己在底层被encoding的信息。如下图，预测T2的时候，T2的第二个隐层会接受到所有Ti第一个隐层传递来的消息(例如T1)，这个消息已经包含了它自己，因为E2被编码进第一层的所有隐层。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/bert_seeself.png)

MLM的具体方法是从输入序列中随机选取15%的单词，标记为**"[MASK]"标识符** (80%概率)，替换成其他随机单词 (10%概率)，或者保持不变 (10%概率)，然后**整个样本的优化目标从LM替换成只预测这些被选中单词的输出 (为它自己)**，并最小化交叉熵。这样做的目的是既可以双向编码，又可以缓解“自己看见自己”的问题，但注意实际上不是严格的LM优化目标。

* **预训练任务2：NSP (Next Sentence Prediction)**。文中指出，nlp任务如QA, NLI(Natural Language Inference)需要模型理解句子之间的关系，而单纯训练LM是不能捕捉这种关系的，所以作者在预训练加入NSP任务。样本包括A/B这样的句子对，其中正负样本(表示B是否为A的下一个句子)各占50%。利用"[CLS]"输出进行分类。

MLM和NSP训练结束后，参数作为微调下游任务的初始化参数。

**其他要注意的细节**：

* (token/segment/positional) Embedding的初始化，参考[资料](https://www.cnblogs.com/d0main/p/10447853.html)。


**Bert应用到下游任务的两种方法**：
(1) feature extraction
(2) fine-tuning

**几点实验结果**：

(1) 在GLUE上(多为分类任务)发现bert (large)在小数据上比bert (small)表现好, 但由于bert (large)在小数据上微调发挥不稳定，所以作者指出每次都随机初始化然后选出在dev数据集上最好的模型;

(2) 作者比较了**把bert作为特征抽取器**以及**微调bert**两种方法, 并且在CoNLL-2003数据集上做了ner任务。其中作为特征抽取器还抽取了不同层 (embeddings, 倒数第一层，倒数第二层)以及不同层的平均 (最后四层加权和，12层的加权和，最后四层的拼接)的特征，发现这些方法都能达到90%以上的正确率；如果微调bert，更可以把正确率提高到96%以上;

(3) 文中给出**微调**实验设置：finetune的**epoch一般都在3**左右，**lr设置为e-5**量级(例如2e-5), **batch_size为16或者32**。

(4) 实验指出更大的模型往往能取得更好的效果，比如增大hidden_dim，加深encoder layers，增多attention heads, 特别当下游任务的训练数据集很小的时候。

(5) 预训练剔除NSP会让NLI的任务正确率变低，但个人感觉效果只是略微差了1%，看不到NSP的绝对优势。



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


