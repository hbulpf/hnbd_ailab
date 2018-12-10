# 隐马尔科夫模型（HMM）

## 要求知识
### 1. [马尔科夫链](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)
马尔科夫状态：下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关。

## 背景

## 思想/原理
在基于马尔科夫链模型的基础上，将原本的可以直接观察到的状态作为隐藏状态q，增加能够直接观测到的状态作为观测状态v。每个隐藏状态之间能够互相转换，其转换概率构成状态转移矩阵A。每个隐藏状态能够生成观测状态，其生成概率构成观测概念矩阵B。这两个矩阵和初始状态概率向量π构成隐马尔可夫模型λ={A,B,π}。

## 基本方法
以下讨论最基本的问题，求特定序列O{o1,o2,...,ot}的出现概率
### 1.暴力求解
直接针对特定序列O{o1,o2,...,ot}求出现概率。<br>
<img src="https://latex.codecogs.com/gif.latex?P(O)" title="P(O)" /><br>
<img src="https://latex.codecogs.com/gif.latex?=\sum_{H}^{&space;}P(O,H)" title="=\sum_{H}^{ }P(O,H)" /><br>
<img src="https://latex.codecogs.com/gif.latex?=\sum_{H}^{&space;}(P(O|H)P(H)))" title="=\sum_{H}^{ }(P(O|H)P(H)))" /><br>
<img src="https://latex.codecogs.com/gif.latex?=\sum_{H}^{&space;}[(b_{h1}(o_{1})b_{h2}(o_{2})...b_{ht}(o_{t}))(\pi&space;_{h1}a_{h1h2}...a_{ht-1}a_{ht})]" title="=\sum_{H}^{ }[(b_{h1}(o_{1})b_{h2}(o_{2})...b_{ht}(o_{t}))(\pi _{h1}a_{h1h2}...a_{ht-1}a_{ht})]" /><br>
H代表所有隐藏序列的集合。
### 2.前向求解
通俗来讲就是从单个观测状态{o1}的出现概率开始逐步推导，到{o1,o2}，直到{o1,o2,...ot}。<br>
首先将当前隐藏状态为qi时序列为{o1,o2,...ot}称作<img src="https://latex.codecogs.com/gif.latex?a_{t}(i)" title="a_{t}(i)" />，因此<img src="https://latex.codecogs.com/gif.latex?a_{1}(i)=\pi_{i}b_{i}(o_{1})" title="a_{1}(i)=\pi_{i}b_{i}(o_{1})" /><br>
因此<img src="https://latex.codecogs.com/gif.latex?a_{2}(i)=&space;[\sum_{j=1}&space;^{N}a_{1}(j)q_{ji}]b_{i}(2)" title="a_{2}(i)= [\sum_{j=1} ^{N}a_{1}(j)q_{ji}]b_{i}(o_{2})" /><br>
得出<img src="https://latex.codecogs.com/gif.latex?a_{t}(i)=&space;[\sum_{j=1}&space;^{N}a_{t-1}(j)q_{ji}]b_{i}(o_{t})" title="a_{t}(i)= [\sum_{j=1} ^{N}a_{t-1}(j)q_{ji}]b_{i}(o_{t})" /><br>
从而能够逐步推算P(O)。

## 与其他方法对比

## 应用场景
在深度学习的问题上，一般隐马尔可夫用于解决自然语言处理的问题。
### 解释：
接收端如何解析？假设接收端的观测信息为o1,o2,...从所有的源信息中找到最可能产生出观测信息的那一个信息串s1,s2,… 就是信号源发送的信息。即使P(s1,s2,...|o1,o2,...)达到最大值的那个信息串s1,s2,… 。利用贝叶斯变换成： P(o1,o2,...|s1,s2,...)⋅P(s1,s2,...)/P(o1,o2,...)。其中P(o1,o2,...)为可以忽略的常数，因此上面公式等价为P(o1,o2,...|s1,s2,...)⋅P(s1,s2,...)--(a)。 这个公式可以由隐含马尔科夫模型来估计。[[3]](https://blog.csdn.net/ZJL0105/article/details/81591426 )

## 数据集

## 代码实践

## 参考资料
1. [隐马尔科夫链](https://blog.csdn.net/weixin_39337018/article/details/82013089) . https://blog.csdn.net/weixin_39337018/article/details/82013089
2. [一文搞懂HMM（隐马尔可夫模型）](http://www.cnblogs.com/skyme/p/4651331.html) . http://www.cnblogs.com/skyme/p/4651331.html
3. [机器学习笔记6 －－ 隐马尔科夫模型 Hidden Markov Model](https://blog.csdn.net/ZJL0105/article/details/81591426) . https://blog.csdn.net/ZJL0105/article/details/81591426

------------
@  [营养大数据 数据科学组](http://git.quietalk.cn/hnbd/data)      
[@鹏飞](http://git.hnbdata.cn/lipengfei)