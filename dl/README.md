# 深度学习：从算法到实战
## 介绍
本仓库需涵盖深度学习算法和应用实例，算法包括DNN、CNN、RNN/LSTM、GAN及强化学习等，应用实例包括计算机视觉的目标检测、图像生成，自然语言处理的文本自动摘要等，团队同学需了解、理解、掌握深度学习的基础和前沿算法，并拥有深度学习算法实战技能。

## 团队能力
+ 掌握深度学习核心算法技术
+ 掌握面向不用场景任务的深度学习应用技术
+ 熟悉各种不同深度神经网络的拓扑结构及应用
+ 熟悉前沿深度学习强化学习等热点技术，把握深度学习的技术发展趋势
+ 提升解决深度学习实际问题的能力

# 内容大纲
## 一 绪论
+ 人工智能和机器学习概述
    + 人工智能历史和现状
    + 从专家系统到机器学习
+ 深度学习概述
    + 从传统机器学习到深度学习
    + 深度学习历史
    + 深度学习的能与不能

## 二 神经网络基础
+ 浅层神经网络
    * 从生物神经元到单层感知器
    * 多层感知器
    * 反向传播和梯度消失
+ 从神经网络到深度学习
    * 逐层预训练
    * 自编码器和受限玻尔兹曼机
    * Beyond预训练+ 

## 三 卷积神经网络
+ 卷积神经网络绪论
    * 卷积神经网络 vs 传统神经网络
    * 卷积神经网络的基本应用
        - 图像分类 image caption
        - 图像检索 image retrieval
        - 物体检测 object detection
        - 图像分割 image segmentation
        - 图像理解 image caption
    * 应用拓展
        - 自动驾驶 self-driving
        - 人脸识别 face recognition
        - 情感识别 facial expression recognition
        - 动作识别 action recognition
        - 图像生成 image generation
        - 风格转化 style transfer
+ 基本组成结构
    * 卷积
    * 池化
    * 全连接
+ 卷积神经网络典型结构
    * AlexNet
    * ZFNet
    * VGG
    * GoogleNet
    * ResNet
+ 卷积神经网络实战（代码解）
+ 总结

## 四 循环神经网络
+ 循环神经网络的应用
    * 机器翻译 machine translation
    * 语音识别 speech recognition
    * 视觉问答 visual question answering
    * 图像理解 image caption
    * 语音问答 speech question answering
+ 循环神经网络 vs 卷积神经网络
    * 技术
    * 应用场景
+ 循环神经网络的基本结构
    * 实例-智能系统
    * 多种递归结构
    * 深度RNN
    * 双向RNN
    * BPTT算法
+ 循环神经网络的模型变种
    * 传统RNN存在的问题
    * LSTM
    * Grid-LSTM
    * GRU
    * 各模型对比
+ 扩展
    * 其他解决RNN梯度消失的方法
    * 基于注意力机制的RNN （attention-based RNN）
+ 总结 

## 五 目标检测
+ 目标检测绪论
    * 概念
    * 评价准则
    * 数据集
    * 竞赛
+ 目标检测战前准备
    * 滑动窗口
    * 目标候选生成
    * 难样本挖掘
    * 非极大值抑制
    * 检测框回归
+ 目标检测：两阶段方法
    * R-CNN
    * SPP-Net
    * Fast R-CNN
    * Faster R-CNN
    * FPN
    * RFCN
+ 目标检测：单阶段方法
    * YOLO
    * SSD
    * Retina Net
+ 荟萃：目标检测方法对比
+ 10行代码实现目标检测
+ 拓展：视频中的目标检测
+ 总结 

## 六 生成对抗网络GAN基础
+ 生成式对抗网络简介
    * 背景
    * GAN案例
        - 图像生成
        - 图像超像素
        - 图像修复
        - 风格转换
        - 文字生成图片
    * GAN应用
        - 数据增广
        - 迁移学习/领域自适应
        - 无监督特征学习
        - 其他
+ 生成式对抗网络基础
    * 生成式对抗网络（Generative Adversarial Network，GAN）
        - 直观解释GAN
        - 模型和目标函数
        - 全局最优解
        - PyTorch实现
    * 条件生成式对抗网络（Conditional GAN， cGAN）
        - 直观解释cGAN
        - 模型和目标函数
        - PyTorch实现
    * 深度卷积生成式对抗网络（Deep Convolutional GAN，DCGAN）
        - 网络结构
        - PyTorch实现
    * Wasserstein GAN （WGAN）
        - JS距离缺陷
        - Wasserstein距离和Wasserstein损失
        - 模型和目标函数
        - PyTorch实现 

## 七 生成对抗网络GAN前沿与实战
+ 生成式对抗网络前沿
    * ProgressiveGAN
    * Spectral Normalization GAN
    * Self-Attention GAN
+ 生成式对抗网络实战
>以图像翻译为案例，由浅入深实现一个工程
   
    * 用GAN实现图像翻译：Pixel2Pixel
        - U-Net
        - PatchGAN
        - Instance Normalization
        - 详细的Pytorch实现
    * CycleGAN
        - Cycle-Consistent 损失
        - 详细的Pytorch实现
    * StarGAN
        - 多领域图像翻译
        - 详细的Pytorch实现 

## 八 前沿技术
+ 深度强化学习
    * 引言：强化学习相关概念、理论基础、深度强化学习的应用
    * 基于策略的方法：策略梯度法
    * 基于值的方法：Deep Q-Network
    * 两种方法的结合：Actor-Critic方法
    * 深度强化学习劝退？优势与挑战
+ 迁移学习
    * 引言：概念、定义与应用
    * 迁移学习的种类及代表性方法
    * 具化迁移学习：域自适应
    * 迁移学习展望
+ 图神经网络
    * 引言：概念与应用
    * 基于空域的图神经网络方法：以门限图递归神经网络为例
    * 基于频域的图神经网络方法：图卷积神经网络（GCN）
    * 展望
+ 深度学习可视化及解释
    * 可视化神经网路
    * 解锁黑箱模型：在路上
+ 深度学习的未来

## 九 PyTorch入门基础
+ 如何用PyTorch完成实验?
    * 如何加载、预处理数据集？
    * 如何构建我想要的模型？
    * 如何定义损失函数、实现优化算法？
    * 如何构建对比实验（baseline）?
    * 如何迭代训练、加速计算（GPU）、存储模型？
+ 用PyTorch 实现经典模型
    * 计算机视觉经典模型实现
        - 怎么实现VGG？
        - 怎么实现GoogleNet？
        - 怎么实现ResNet？
    * 自然语言处理经典算法实现
        - 怎么实现神经网络语言模型？
        - 怎么实现Sequence to sequence + attention（含有注意力机制的序列建模）?
        - 怎么实现sequence labeling（序列标注模型）?

## 十 PyTorch实战
+ 计算机视觉应用实战： 用PyTorch 实现实时目标检测
    * 什么是目标检测任务？
    * 目标检测的公开数据集解
    * 目标检测的模型解
    * 典型算法与实现
        - YOLO
        - SSD
+ 自然语言处理应用实战：用PyTorch 实现文本自动摘要生成
    * 什么是文本自动摘要生成任务？
    * 文本摘要生成的公开数据集解
    * 文本摘要生成的模型解
    * 典型算法与实现
        - Pointer-generator
        - Fast_abs_rl

# 参考
1. 深度学习:算法到实战 . https://mp.weixin.qq.com/s/8cToybAbanQSi9vgLDJIzA