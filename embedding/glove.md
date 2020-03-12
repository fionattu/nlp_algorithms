## Syllabus

Mark notes related to **Glove** algorithm:

* Comparison between **Glove** and **word2vec or matrix factorization** based algorithms (**LSA and HAL**). 
* Motivation and derivation of Glove.
* Evaluation of word vectors.


## Motivation of Glove

Glove是由斯坦福著名的nlp公开课cs224n的主讲教授Christopher D. Manning和其学生提出的。其通过结合word2vec和早期基于词频的矩阵分解的方法，提出Glove算法，让词向量效果得到比较大的提升。

基于矩阵分解的方法：LSA和HAL等，考虑了global statitistics(count-based), 可以捕捉到词语的相似性，但是在词类比评估上表现不好，似乎只达到词向量空间的局部最优。而且共现词频矩阵很占内存，对于自然语言的更新换代训练代价比较高；而word2vec等方法，摒弃了矩阵的概念，利用词语和其上下文的词的相似度对词向量进行建模，在词向量评估上得到很好的效果，但却没有考虑词语在全部语料的情况，例如上下文某个词相对于中心词在所有语料的共现次数。

Glove是结合了以上两种方法优点的启发式算法: 1. 利用滑动窗口采样训练集，并且统计了词语共现次数(global statitistics); 2. 用词向量的点乘作为词语的相似度(word2vec)。优化目标很简单，就是让词语的相似度无限逼近它们的共现次数。

模型来源于原文作者基于以下现象的观察：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/glove.png)

其中k是ice和steam的上下文词语。第一二行是word2vec算法下k在给定词的概率。如果单单看P(k | ice)或P(k | steam)，我们没办法知道k跟ice或steam到底有多相似。但如果通过概率的对比，我们可以看出，solid-ice远远比solid-steam更相关，因为P(k|ice)/P(k|steam)远远大过1；而gas-steam比gas-ice更相关，因为P(k|steam)/P(k|ice)远远小于1；比值接近1的，不是都相关就是都不相关。这个观察启发了作者：应该用条件概率的比值来建模而不是用概率本身。
 

## Derivation of Glove

Glove相关公式见。。。。

## 词向量的评估

词向量的评估方法分为两种：

* **Intrinsic**
	
	**Word analogy(词类比)方法：**一般是采用queen-king+man=women的方法，在一些公开的评测数据集上进行验证。即给定前三个词向量，看通过词向量运算后的结果向量是不是和第四个词的词向量最接近。
	
	**词相似度方法：**著名的WordSim353数据集，用人工标注的方法定义了词语间的相似度。
	
	
* **Extrinsic**

	单从i词语相似度来评估词向量的方法受到质疑，因为其对语料的变化(来源，大小等等)非常敏感，很多人认为词向量的好坏是task-specific，脱离实际任务的词向量是没有意义的。

	Extrinsic方法是把词向量输入一些具体的有标注的nlp任务，例如文本分类，Ner等，看效果有没有提升。
	
	
	
	
	
	
	
	