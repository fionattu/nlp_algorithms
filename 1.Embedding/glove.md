## Syllabus

Mark notes related to **Glove** algorithm:

* Comparison between **Glove** and **word2vec** or **LSA** (matrix factorization). 
* Motivation of Glove.
* Derivation of Glove.


## Glove vs. LSA vs. Word2vec

Glove是由斯坦福著名的nlp公开课cs224n的主讲教授Christopher D. Manning和其学生提出的。其通过结合word2vec和早期基于词频的矩阵分解的方法(LSA)的优点，提出Glove算法，让词向量效果得到比较大的提升。

### Glove vs. LSA (Latent Semantic Analysis)

基于矩阵分解的方法例如LSA(利用SVD)等，统计了不同文本词语的出现次数（或者词语的tfidf值），并且对矩阵进行分解。这种方法考虑了global statitistics(count-based), 可以捕捉到词语的相似性，但是在词类比评估上表现不好，似乎只达到词向量空间的局部最优。而且矩阵分解计算难度大，矩阵本身也很占内存，对于语料的更新训练代价比较高。

* 两者都是基于共现矩阵在操作
* LSA（Latent Semantic Analysis）可以基于co-occurance matrix构建词向量，实质上是基于全局语料采用SVD进行矩阵分解，然而SVD计算复杂度高
* Glove没有直接利用共现矩阵，而是通过ratio的特性，将词向量和ratio联系起来，建立损失函数，采用Adagrad对最小平方损失进行优化（可看作是对LSA一种优化的高效矩阵分解算法）


### Glove vs. Word2vec

word2vec等方法，摒弃了矩阵的概念，利用词语和其上下文的词的相似度对词向量进行建模，在词向量评估上得到很好的效果，但却没有考虑词语在全部语料的情况，例如上下文某个词相对于中心词在所有语料的共现次数。

* Word2vec是局部语料库训练的，其特征提取是基于滑窗的；而glove的滑窗是为了构建co-occurance matrix（上面详细描述了窗口滑动的过程），统计了全部语料库里在固定窗口内的词共线的频次，是基于全局语料的，可见glove需要事先统计共现概率；因此，word2vec可以进行在线学习，glove则需要统计固定语料信息。
* Word2vec是无监督学习，同样由于不需要人工标注，glove通常被认为是无监督学习，但实际上glove还是有label的，即共现次数log(X_i,j)
* Word2vec损失函数实质上是带权重的交叉熵，权重固定；glove的损失函数是最小平方损失函数，权重可以做映射变换。
* Glove利用了全局信息，使其在训练时收敛更快，训练周期较word2vec较短且效果更好。


**总结一下：Glove是结合了以上两种方法优点的启发式算法: 1. 利用滑动窗口采样训练集，并且统计了词语共现次数(global statitistics); 2. 用词向量的点乘来模拟词语的相似度(word2vec)。优化目标很简单，就是让词语的相似度无限逼近它们的共现次数。他们的共同特点是不考虑词语的顺序。**


## Motivation of Glove

模型来源于原文作者基于以下现象的观察：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/glove.png)

其中k是ice和steam的上下文词语。第一二行是word2vec算法下k在给定词的概率。如果只看条件概率P(k | ice)或P(k | steam)，我们无法衡量k跟ice或steam到底有多相似。但如果通过概率的对比，我们可以看出，词对solid-ice比solid-steam更相关，因为P(k|ice)/P(k|steam)远远大过1；而gas-steam比gas-ice更相关，因为P(k|steam)/P(k|ice)远远小于1；比值接近1的，都相关或者都不相关。这个观察启发了作者：应该用条件概率的比值来建模而不是用概率本身。
 

## Derivation of Glove

<a href="https://github.com/fionattu/nlp_algorithms/blob/master/pics/derivation/glove.pdf" target="_blank" rel="noopener">Glove目标函数推导</a>。推导过程非常开脑洞，基于启发式，不是因果关系。


	
	
	
	
	
	
	
	