## Syllabus

Mark notes related to:

* Subword Models
* Pre-trained Concepts (fine-tuning/network freezing) in LM: TagLM, Elmo, ULMfit
* Transformers: Attention is all you need
* Bert

## Subwords Model

### Motivation


* OOV Problem
* Morphology

### Subword Model Basics

Pure char-level NMT 训练特别慢

### Two Trends

* Word pieces: 
 	1. BPE 
 	2. Wordpiece/Sentencepiece of Google/Bert
 	3. learning word representation from char level
* Hybrid 

### Other Inspiration from Char-based Model: FastText Embeddings

## Transformers

在Transformer问世之前，[加入attention机制的seq2seq模型 (encoder-decoder)](https://github.com/fionattu/nlp_algorithms/blob/master/3.Seq2seq_attention/seq2seq_attention.md)在各个任务上已经取得了很大的提升，但rnn模型的顺序处理无法并行化，使得它的训练过程特别耗时；LSTM等门机制也只能缓解rnn的长依赖问题。rnn的优点是能考虑上下文信息对句子进行编码，既然attention这么强大且能自己学习上下文的权重，如果再加入词语顺序（order）的信息，是否也能取代rnn呢？2017年google发表了论文“Attention is all you need”，在机器翻译上取得了很好的效果 (英德翻译中，BLEU提升2）。他们在论文中提出transformer模型，后来一举成名的BERT也是基于transformer构建的。


**References:** 

* Transformer论文精读：Attention is all you need

* [用Jupyter一步步实现Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

* [外国博主Jay Alammar对前沿nlp模型ELMo/transformer/bert的详解](http://jalammar.github.io)

### Architecture

论文算法模型，依旧遵循encoder-decoder的结构。按照论文的设置，encoder和decoder均为6个transformer组成，最底层输入是word embedding(dim=512)，之后每一层的输入都为上一层的输出, dim均为512。如下图，左边为encoder，右边是decoder。

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

原文提出的Multi-head Attention是重复执行8个One-head Attention。首先我们来了解One-head Attention的流程。

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/self_attention_0.png)

![images](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/self_attention_1.png)

**Multi-head Attention**

### Encoder: Positional Encoding 

### Decoder: Encoder-Decoder Attention



* Mask

### Comparison with RNN-based Model



## More
* tensor2tensor
* Residual Connection

* image/music transformer


