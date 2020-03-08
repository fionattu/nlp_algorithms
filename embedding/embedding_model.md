## Syllabus

Mark notes related to:

* Word2Vec algorithms: Skip-gram and CBOW. 
* Tricks that proposed to accelerate learning: Negative Sampling, Hierachical Softmax.


## Word2Vec Algorithms

word2vec目的是将**词语转化为向量**，以便进行自然语言更high-level的任务：文本分类，机器翻译，文本摘要，对话系统等。本质是用“**单词和其在自然语言句子中的邻居词（context上下文）的相似性”**对算法进行建模。通过调整中心词和周围词的词向量来最大化条件概率，即给定中心词预测其周围词出现的概率（or vice versa）。

在word2vec出来前，词表示还有one-hot embedding和SVD方法，但他们都有比较大的弊端：one-hot embedding属于**稀疏表示**，维度大，正交性导致不利于计算词语间的相似度；SVD方法属于**稠密表示**(或**分布式表示**)，但其用于矩阵分解的初始输入矩阵会随着新词的加入平方级别地增长，新加入语料也需要重新计算该矩阵，分解过程计算起来也比较困难。

word2vec特指skip-gram和CBOW模型，来自以Tomas Mikolov为代表的google学者们这篇论文：Efficient Estimation of Word Representations in Vector Space。

word2vec属于自编码无监督学习，语料来自于文本，不需要用户标注，只需要设置滑动窗口（默认为5，即上下文邻居各2个）来采样训练语料。其神经网络结构只有三层：输入输出为词语的one-hot encoding, 中间隐层实际是输入词的词向量并且没有激活函数，输出层的激活函数是softmax函数，用于归一化，以便得到所有词在给定词下的概率分布。优化目标是最小化输出的概率分布和真实数据的**交叉熵**。最终学到的是输入层到隐层的权重矩阵W，即词向量矩阵。

其实隐层到输出层也有一个词向量矩阵W'，但一般大家用的是W。或许可以通过拼接或者平均的方法组成词向量，看哪个效果好用哪个。

### Skip-gram
用中心词预测周围词。一个长度为L+1滑动窗口对应L个训练样例。输入的one-hot表示中心词位置为1，输出的one-hot表示周围词对应位置为1。
	
![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/skipgram.png) 

### CBOW (continuous bag of words)

用周围词预测中心词。一个长度为L+1滑动窗口对应1个训练样例。输入的one-hot表示所有周围词位置为1，输出的one-hot表示中心词对应位置为1。隐层会对所有周围词的词向量进行加和并且平均。

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/skipgram.png)
	
目标函数(objective function)和梯度下降的公式推导见<a href="https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/w2v公式推导.pdf/" target="_blank">w2v公式推导</a>

## Word2Vec Tricks

softmax方法每一个训练语料都要进行一次分母计算 $\sum_{w=1}^{V} exp(u_wv_c)$。而语料中的$V$单位为millions，导致计算相当耗时。

Tomas Mikolov为代表的google学者们继而提出第二篇论文：Distributed Representations of Words and Phrases and their Compositionality，用来加速训练过程。其中主要涉及三个tricks。

### Hierachical Softmax (层次softmax)
用哈夫曼树(二叉树)取代softmax层。哈夫曼树的叶子节点为$V$个词汇，在文本中频率越高的单词离根节点越近。而非叶子节点均用向量表示，是这个模型需要进行学习的参数。
	
![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/hierachical_softmax.png)
  
隐藏层的结果$h$直接输入哈夫曼树的根节点，并与根节点以及到目标词的路径上所有非叶子节点进行点乘并过一次sigmoid函数。因此softmax被$logV$个二分类器代替(前向计算复杂度从$V$减少为$logV$)。如果当前节点到目标叶子节点往左走(默认分类为1)，该节点的输出为二分类器的输出$\sigma$, 否则输出为$1-\sigma$。如果路径为左-右-右，经过的节点分别为$w_1$-$w_2$-$w_3$, 目标函数是最大化$\sigma(h*w_1)*(1-\sigma(h*w_2))*(1-\sigma(h*w_3))$, 转化为最小化负log loss并对$w_1$,$w_2$,$w_3$进行梯度下降。其余不在路径上的非叶子节点不需要参与更新，大大减少了计算量。反向传播复杂度也从$V$减少为$logV$。
  
### Negative Sampling (负采样)

原生的softmax层考虑了所有词语，负采样只考虑一个正样本和几个负样本，并对每个样本进行一次二分类（逻辑回归），这样输出的神经元可以从百万到几十个，权重矩阵也大大减少。原来的训练样本只考虑了词语的共现关系，但负采样需要产生一些错误的词对，来构成**错误的共现关系**。

通过负采样方法，word2vec的目标函数分别为：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/objfunc_negsampling.png)
  
如何进行采样呢？如下图，每个单词被负采样的频率取决于它在语料中出现的频次。其中$3/4$是基于经验所得，相比于$y=x$,可以提高一些频次较低的单词被采样的概率，并降低频次较高的单词的采样概率。

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/negative_sampling_1.png)

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/negative_sampling_2.png)


  
### Subsampling Frequent Words (二次抽样高频词)
	
目的是防止一些高频词频繁出现在训练样本，比如the。每个单词用它出现的频率和总单词个数的比值，输入到采样函数中得到它的二次抽样概率，表示其被保留的概率，频率越高，保留概率越低。

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/subsampling_frequent_words_1.png)

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/embedding/pics/subsampling_frequent_words_2.png)

 
## More

* 比较skip-gram和CBOW的效果

* 哈夫曼编码

