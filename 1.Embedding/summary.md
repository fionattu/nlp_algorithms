
## Embedding Models
* <a href="https://github.com/fionattu/nlp_algorithms/blob/master/1.Embedding/word2vec.md" target="_blank" rel="noopener">Word2vec: skipgram, CBOW</a>
* <a href="https://github.com/fionattu/nlp_algorithms/blob/master/1.Embedding/glove.md" target="_blank" rel="noopener">Glove</a>

## References

* 腾讯AI实验室词向量
* 论文精读
	1. Word2Vec 2013 of Google: [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
	2. Glove 2014: [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162.pdf) 
	3. FastText 2016 of Facebook Fair: [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf) 
	4. DSG 2018 of Tencent AI Lab: [Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings](https://www.aclweb.org/anthology/N18-2028.pdf)

## 中文预训练词向量

### 上百种不同语料的预训练中文词向量
北京师范大学和中国人民大学合作并开源的[Github地址](https://github.com/Embedding/Chinese-Word-Vectors)。他们抓取了不同语料：百度百科，中文维基百科，人民日报，搜狗新闻，金融新闻，知乎问答，微博，文学作品等，来训练不同领域的词向量。**这些语料大小在1-6G，单词总数在百万量级。他们使用HanLP进行分词，采用带负采样优化的skip-gram模型来训练词的稠密表示(d=300)以及使用PPMI(基于SVD)的方法来训练词的稀疏表示**。

### 腾讯AI实验室开源词向量
腾讯在2018年开源了一个包含800多万中文词汇词向量的数据集，在内部任务像对话回复，医疗实体识别等业务应用中取得显著提升，[下载地址](https://ai.tencent.com/ailab/nlp/embedding.html)。训练语料是跨领域的，来自新闻，网页和小说，可以自动发现新词，并且他们自称在覆盖率(“不念僧面念佛面”、“冰火两重天”、“煮酒论英雄”)，新鲜度(“恋与制作人”、“三生三世十里桃花”、“打call”)和准确性(相似词检索结果)上进行了提升。使用了自研的[Directional Skip-Gram (DSG)](https://www.aclweb.org/anthology/N18-2028.pdf)算法进行训练，词向量为200维。

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

