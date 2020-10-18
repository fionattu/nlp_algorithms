**Fundamentals of Neural Networks**, including basic concepts and practical tricks.

## Neurons(神经元)

Neurons(神经元)是神经网络的最小组成部分。数据通过神经元的激活函数（activation function）进行非线性变换，而神经网络不同层之间的矩阵乘法均为线性变换，所以激活函数的存在很重要。

### 激活函数

ReLU > tanh > sigmod (ReLU训练快且效果好，被广泛采用)，见[激活函数](https://github.com/fionattu/nlp_algorithms/blob/master/pics/derivation/activation_functions.pdf)。

## BP反向传播

以NER任务为例子，推导Elemental BP和vectorized BP。参考[BP推导](https://github.com/fionattu/nlp_algorithms/blob/master/pics/derivation/back_prop.pdf)。

## Overfitting(过拟合)

神经网络在样本数据少，模型参数太多的情况下容易导致过拟合。**判断过拟合的标准，是出现训练集正确率较高/损失函数较小然而在测试集上预测正确率较低/损失函数较大的现象**。防止过拟合并提高模型泛化性能，可以引入regularization和dropout的方法。

### Regularization(正则化)

正则化是在损失函数加入一个权重可控($\lambda$)的**L2范数**优化项，格式如下：

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/regularization_loss.png)

其中，L2范数是参数的平方和再开方。其作用是充当惩罚项，防止训练出来的参数太大，导致模型不稳定(输入稍微变化就可以引起结果的大波动)。$\lambda$的选择很重要: $\lambda$太大，容易导致学习出来的权重太小，模型很难学到有用的特征；$\lambda$太小，则失去正则化的效果，所以需要进行hyperparameter-tunning。

![image](https://raw.githubusercontent.com/fionattu/nlp_algorithms/master/pics/L2_norm.png)


注意偏置项不需要加入正则化，其与输入的特征数值无关，仅仅是在输出的结果加一个位移；对偏置项$b$进行正则化反而会让效果变差, 因为权重$w$都偏小，会让结果无法区别。可以参考[why-is-the-bias-term-not-regularized-in-ridge-regression](why-is-the-bias-term-not-regularized-in-ridge-regression)。

### Dropout

Dropout是一个实际训练时可以灵活选择的trick。Dropout的工作机制：在前向传播过程中，让神经元以1-p的概率失效，输出0：反向传播时，只传递梯度给活跃的神经元，与失效神经元连接的参数不参与更新。在测试数据集上，不使用dropout。

为什么使用dropout可以防止过拟合，大概有这两方面的原因：1) 其效果相当于每次都训练部分神经网络，总体来看是对不同神经网络结果取平均；2）减少神经元之间的相互关系（即某个神经元需要依赖另一个才能运作），减少模型对局部特征的依赖。

## 数据处理

### 特征值
最常用的是mean subtraction，让所有特征能zero-center。平均值是训练集的平均值，测试集验证集预测的时候都要将数据减去这个平均值，不另外计算自己的平均值。

### 参数
实验证明采用正态分布效果较好。

## Optimizers

优化器主要涉及以下两种不同的优化策略：

**1）自适应学习率(learning rate)**：学习率决定了每次参数更新的步长，最终决定训练的速度。学习率太大，模型容易跳过全局最小值；学习率太小又会让训练过程花费太多时间，每次参数只有很小的调整，也很容易陷入局部最小点。有个手动设置学习率变化的方法叫annealing: 推荐大家从一个比较高的学习率开始然后慢慢降低学习率。一开始离全局最小值还较远，较大的学习率可以加快收敛；最后到全局最小值附近的时候，较小的学习率可以让损失函数慢慢逼近全局最优。除了让学习率随着时间指数减少: $a(t) = a_0\exp(-kt)$; 还有一个方法是: $a(t) = a_0\tau/max(t, \tau)$, 即设置一个时间点$\tau$, 让学习率开始减少。后者在实验中效果表现很好。

**2）动量(momentum)法**：根据之前所有的梯度更新(方向和数值)决定下一步的更新策略。方法如下：

```
# compute a standard momentum update on parameter x
v = mu*v - alpha*grad_x
x += v
```

其中mu是个超参，代表动量因子，mu*v积累了之前的梯度，相当于加权平均了之前的梯度和当前的梯度。特点：

* 下降初期时，使用上一次参数更新，下降方向一致，乘上较大的mu能够进行加速
* 下降中后期时，在局部最小值来回震荡的时候，mu使得更新幅度增大，跳出陷阱
* 在梯度改变方向的时候，mu能够减少更新 

**总而言之，momentum项能够在相关方向加速SGD，抑制振荡，从而加快收敛。**



下面介绍经典的优化方法：

### AdaGrad - 自适应的学习率

```
# Assume the gradient dx and parameter vector x 
cache += dx**2
x += -learning_rate * dx / np.sqrt(cache + eps) ## eps防止分母为0
```

* 前期dx较小的时候，np.sqrt(cache + 1e-8)较小，能够放大梯度
* 后期dx较大的时候，np.sqrt(cache + 1e-8)较大，能够约束梯度
* 适合处理稀疏梯度：稀疏梯度由于积累的梯度分母项较小，相比于其他梯度能被放大，从而加速收敛
* **缺点：中后期，分母上梯度平方的累加将会越来越大，使梯度->0，使得训练提前结束**

### RMSProp - AdaGrad的变种

```
# Update rule for RMS prop
cache = decay_rate * cache + (1 - decay_rate) * dx**2 
x += -learning_rate * dx / (np.sqrt(cache) + eps)
```

Hinton在论文中使用decay_rate=0.9。解决了AdaGrad梯度急剧下降的问题。Adagrad会累加之前所有的梯度平方，而RMSprop仅仅是计算对应的平均值，因此可缓解Adagrad算法学习率下降较快的问题。
	
### Adam - RMSProp + 动量

```
# Update rule for Adam
m = beta1*m + (1-beta1)*dx  # 动量
v = beta2*v + (1-beta2)*(dx**2) # RMSProp
x += -learning_rate * m / (np.sqrt(v) + eps)
```
**实践证明，Adam的表现比其他适应性方法要好。**


## 其他

* 梯度下降原理，参考泰勒展开式，非常好的资料: [参考资料](https://blog.csdn.net/pengchengliu/article/details/80932232)

* L1正则化：L1 regularization, which sums over the absolute values (rather than squares) of parameter elements – however, this is less commonly applied in practice since it leads to sparsity of parameter weights.

* Batch Normalization： 保证测试集上神经元的输出和训练集的在同个可接受范围内。