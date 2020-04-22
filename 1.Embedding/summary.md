
## Embedding Models
* <a href="https://github.com/fionattu/nlp_algorithms/blob/master/1.Embedding/word2vec.md" target="_blank" rel="noopener">Word2vec: skipgram, CBOW</a>
* <a href="https://github.com/fionattu/nlp_algorithms/blob/master/1.Embedding/glove.md" target="_blank" rel="noopener">Glove</a>

## References

* 腾讯AI实验室词向量
* 论文精读
	1. Word2Vec 2013: [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
	2. Glove 2014: [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162.pdf) 
	3. FastText 2016: [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf) 


## 词向量的评估

词向量的评估方法分为两种：

### Intrinsic
	
**Word analogy(词类比)方法:** 一般是采用queen-king+man=women的方法，在一些公开的评测数据集上进行验证。即给定前三个词向量，看通过词向量运算后的结果向量是不是和第四个词的词向量最接近。
	
**词相似度方法:** 著名的WordSim353数据集，用人工标注的方法定义了词语间的相似度。
	
	
### Extrinsic

单从词语相似度来评估词向量的方法受到质疑，因为其对语料的变化(来源，大小等等)非常敏感，很多人认为词向量的好坏是task-specific，脱离实际任务的词向量是没有意义的。

Extrinsic方法是把词向量输入一些具体的有标注的nlp任务，例如文本分类，Ner等，看效果是否有提升。

## 词向量效果分析
词向量的效果与训练语料的来源和特点有关。用gensim提供的词向量来做分析，根据余弦相似度来计算一个词向量最相似的topk个词向量，发现有以下问题：

* 多义词：topk里面应该出现这个词的多义词，比如输入*lie*，topk会出现*lies, sit, lay*等词；但对于*right，star*等词并没有出现多义词，这说明训练词向量的语料不够丰富，只涵盖了词语的部分含义；

* 同义词，反义词：出现一个词与同义词的相似度 **<** 其与反义词的相似度，例如：sim(bad, evil) **<** sim(bad, good)。语料来自于google新闻，可推测反义词的上下文会比较像，表示对一件事情的不同观点；相反的，不会用多个同义词来表述同个观点，于是会出现这个情况。

