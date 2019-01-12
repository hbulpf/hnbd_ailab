Image captioning 荟萃
基础入门
李飞飞：如何教计算机理解图片 一个曾经爆火的TED演讲，看完也就知道 image captioning是要干嘛了 - 简介：小孩看到图时，能立刻识别出图上的简单元素，例如猫、书、椅子。现如今，计算机也拥有足够智慧做到这一点了。接下来呢？斯坦福大学的计算机视觉专家李飞飞将描绘当今人工智能科技的前沿领域。她和她的团队建立起了一个含有1500万张照片的数据库，并通过该数据库来教计算机理解图片。[https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=zh-cn] [http://open.163.com/movie/2015/3/Q/R/MAKN9A24M_MAKN9QAQR.html]
梅涛：深度学习为视觉和语言之间搭建了一座桥梁
◦[http://www.msra.cn/zh-cn/news/features/vision-and-language-20170713]
看图说话的AI小朋友——图像标注趣谈(上，下)
◦[https://zhuanlan.zhihu.com/p/22408033]
◦[https://zhuanlan.zhihu.com/p/22520434]
李理：从Image Caption Generation理解深度学习（part I，II, III）
◦[http://www.easemob.com/news/739]
◦[http://www.easemob.com/news/740]
◦[http://www.easemob.com/news/741]
Multimodal —— 看图说话（Image Caption）任务的论文笔记（一）评价指标和NIC模型
Multimodal —— 看图说话（Image Caption）任务的论文笔记（二）引入attention机制
◦[http://www.cnblogs.com/Determined22/p/6910277.html]
◦[http://www.cnblogs.com/Determined22/p/6914926.html]
图片语义分析 腾讯CDG社交与效果广告部（广点通）
◦[http://www.flickering.cn/ads/2015/02//语义分析的一些方法三]
“无中生有”计算机视觉探奇 云栖社区-阿里云
◦[https://yq.aliyun.com/articles/85266]
Show and Tell: image captioning open sourced in TensorFlow
◦[https://research.googleblog.com/2016/09/show-and-tell-image-captioning-open.html]

# 进阶论文
2013
Generating Natural-Language Video Descriptions Using Text-Mined Knowledge, AAAI 2013. ◦[https://www.aaai.org/ocs/index.php/AAAI/AAAI13/paper/view/6454/7204]

2014
m-RNN模型《 Explain Images with Multimodal Recurrent Neural Networks》 2014 2014年10月，百度研究院的毛俊骅和徐伟等人在arXiv上发布论文，提出了multimodal Recurrent Neural Network（即m-RNN）模型，创造性地将深度卷积神经网络CNN和深度循环神经网络RNN结合起来，用于解决图像标注和图像和语句检索等问题。可以说是image captioning开创性的工作，文中提到“To the best of our knowledge, this is the first work that incorporates the Recurrent Neural Network in a deep multimodal architecture.”
◦[https://arxiv.org/pdf/1410.1090.pdf]
NIC模型 《Show and Tell: A Neural Image Caption Generator》2014 2014年11月谷歌的Vinyals的论文，相较于百度的m-RNN模型，NIC模型的主要不同点在于抛弃RNN，使用了更复杂的LSTM；CNN部分使用了一个比AlexNet更好的卷积神经网络并加入batch normalization； CNN提取的图像特征数据只在开始输入一次。
◦[https://arxiv.org/pdf/1411.4555.pdf]
MS Captivator From captions to visual concepts and back 2014 2014年11月微软提出利用多实例学习，去训练视觉检测器来提取一副图像中所包含的单词，然后学习一个统计模型用于生成描述。对于视觉检测器部分，由于数据集对图像并没有准确的边框标注，并且一些形容词、动词也不能通过图像直接表达，所以本文采用Multiple Instance Learning(MIL)的弱监督方法，用于训练检测器。
◦[https://arxiv.org/pdf/1411.4952.pdf]
Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel, Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, arXiv:1411.2539, NIPS 2014 deep learning workshop.
◦[https://arxiv.org/abs/1411.2539]
Sequence to sequence learning with neural networks. NIPS 2014.
◦[https://arxiv.org/abs/1409.3215]
Large-scale video classification with convolutional neural networks. CVPR, 2014.
◦[http://cs.stanford.edu/people/karpathy/deepvideo/]

2015
Show, Attend and Tell: Neural Image Caption Generation with Visual Attention 2015 2015年Yoshua Bengio等人受注意机制在机器翻译中发展的启发，提出了在图像的卷积特征中结合空间注意机制的方法，然后将上下文信息输入到encoder-decoder框架中。在encoder阶段，与之前直接通过全连接层提取特征不同，作者使用较低层的卷积层作为图像特征，其中卷积层保留了图像空间信息，然后结合注意机制，能够动态的选择图像的空间特征用于decoder阶段。在decoder阶段，输入增加了图像上下文向量，该向量是当前时刻图像的显著区域的特征表达。
◦[https://arxiv.org/pdf/1502.03044.pdf]
Guiding Long-Short Term Memory for Image Caption Generation 2015 使用语义信息来指导LSTM在各个时刻生成描述。由于经典的NIC模型，只是在LSTM模型开始时候输入图像，但是LSTM随着时间的增长，会慢慢缺少图像特征的指导，所以本文采取了三种不同的语义信息(分别是Retrieval-based guidance (ret-gLSTM), Semantic embedding guidance(emb-gLSTM) ,Image as guidance (img-gLSTM))，用于指导每个时刻单词的生成。
◦[https://arxiv.org/pdf/1509.04942.pdf]
Long-term Recurrent Convolutional Networks for Visual Recognition and Description, CVPR 2015.
◦[https://arxiv.org/abs/1411.4389]
Translating Videos to Natural Language Using Deep Recurrent Neural Networks, NAACL-HLT, 2015. ◦[https://arxiv.org/abs/1412.4729]
Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation, CVPR 2015
◦[https://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf]
Phrase-based Image Captioning, arXiv:1502.03671 / ICML 2015
◦[https://arxiv.org/abs/1502.03671]
Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images, ICCV 2015.
◦[https://arxiv.org/abs/1504.06692]
Exploring Nearest Neighbor Approaches for Image Captioning, 2015.
◦[https://arxiv.org/abs/1505.04467]
Language Models for Image Captioning: The Quirks and What Works, Jacob Devlin, Hao Cheng, Hao Fang, Saurabh Gupta, Li Deng, Xiaodong He, Geoffrey Zweig, Margaret Mitchell, ACL 2015.
◦[https://arxiv.org/abs/1505.01809]
Image Captioning with an Intermediate Attributes Layer, 2015.
◦[https://arxiv.org/abs/1506.01144v1]
Learning language through pictures, 2015.
◦[https://arxiv.org/abs/1506.03694]
Univ. Montreal Kyunghyun Cho, Aaron Courville, Yoshua Bengio, Describing Multimedia Content using Attention-based Encoder-Decoder Networks, 2015.
◦[https://arxiv.org/abs/1507.01053]
Cornell Jack Hessel, Nicolas Savva, Michael J. Wilber, Image Representations and New Domains in Neural Image Captioning, 2015.
◦[https://arxiv.org/abs/1508.02091]
Learning Query and Image Similarities with Ranking Canonical Correlation Analysis, ICCV, 2015
◦[https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Learning_Query_and_ICCV_2015_paper.pdf]
Deep Captioning with Multimodal Recurrent Neural Networks (m- RNN) 2015.
◦[https://arxiv.org/abs/1412.6632]
Deep Visual-Semantic Alignments for Generating Image Descriptions TPAMI 2015.
◦[https://arxiv.org/abs/1412.2306]
Sequence to Sequence -- Video to Text. 2015.
◦[https://arxiv.org/abs/1505.00487]

2016
What Value Do Explicit High Level Concepts Have in Vision to Language Problems? 2016 不同于将图像特征经过CNN后直接扔进RNN并，本文作者提出将图像特征用高等级的语义概念表达后再输入RNN会更好。作者首先利用VggNet模型在ImageNet数据库进行预训练，然后进行多标签数训练。给一张图片，首先产生多个候选区域，将多个候选区域输入CNN产生多标签预测结果，然后将结果经过max pooling作为图像的高层语义信息，最后输入到LSTM用于描述的生成
◦[https://arxiv.org/pdf/1506.01144.pdf]
Watch What You Just Said: Image Captioning with Text-Conditional Attention 2016 该模型首先利用卷积神经网络提取图像特征，然后结合图像特征和词嵌入的文本特征作为gLSTM的输入。由于之前gLSTM的guidance都采用了时间不变的信息，忽略了不同时刻guidance信息的不同，而作者采用了text-conditional的方法，并且和图像特征相结合，最终能够根据图像的特定部分用于当前单词的生成。
◦[https://arxiv.org/pdf/1606.04621.pdf]
◦[https://github.com/LuoweiZhou/e2e-gLSTM-sc]
Video Captioning with Transferred Semantic Attributes 2016.
◦[https://arxiv.org/abs/1611.07675]
DenseCap: Fully Convolutional Localization Networks for Dense Captioning, CVPR 2016.
◦[https://arxiv.org/abs/1511.07571]

2017
Self-critical Sequence Training for Image Captioning, 2017 CVPR IBM Watson 研究院发表的这篇论文直接优化了 CIDEr 评价标准（Consensus-based image description evaluation）。由于此目标函数不可微，论文中借鉴基础的强化学习算法 REINFORCE 来训练网络。 该文提出了一个新的算法 SCST（Self-critical Sequence Training），将贪婪搜索（Greedy Search ）结果作为 REINFORCE 算法中的基线（Baseline），而不需要用另一个网络来估计基线的值。这样的基线设置会迫使采样结果能接近贪婪搜索结果。在测试阶段，可直接用贪婪搜索产生图像描述，而不需要更费时的集束搜索（又名定向搜索，Beam Search）。除了 SCST，此论文也改进了传统编码器 - 解码器框架中的解码器单元，基于 Maxout 网络，作者改进了 LSTM 及带注意力机制的 LSTM。综合这两个改进，作者提出的方法在微软的图像描述挑战赛 MS COCO Captioning Challenge 占据榜首长达五个月，但目前已被其他方法超越。
◦[https://arxiv.org/pdf/1612.00563.pdf]
Deep Reinforcement Learning-based Image Captioning with Embedding Reward 2017 cvpr 由 Snapchat 与谷歌合作的这篇论文也使用强化学习训练图像描述生成网络，并采用 Actor-critic 框架。此论文通过一个策略网络（Policy Network）和价值网络（Value Network）相互协作产生相应图像描述语句。策略网络评估当前状态产生下一个单词分布，价值网络评价在当前状态下全局可能的扩展结果。这篇论文没有用 CIDEr 或 BLEU 指标作为目标函数，而是用新的视觉语义嵌入定义的 Reward，该奖励由另一个基于神经网络的模型完成，能衡量图像和已产生文本间的相似度。在 MS COCO 数据集上取得了不错效果。
◦[https://arxiv.org/abs/1704.03899]
Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning 2017 cvpr 弗吉尼亚理工大学和乔治亚理工大学合作的这篇论文主要讨论自适应的注意力机制在图像描述生成中的应用。在产生描述语句的过程中，对某些特定单词，如 the 或 of 等，不需要参考图像信息；对一些词组中的单词，用语言模型就能很好产生相应单词。因此该文提出了带有视觉哨卡（Visual Sentinel）的自适应注意力模型，在产生每一个单词的时，由注意力模型决定是注意图像数据还是视觉哨卡。
◦[https://arxiv.org/pdf/1612.01887.pdf]
◦[https://github.com/jiasenlu/AdaptiveAttention]
Boosting Image Captioning with Attributes, ICCV 2017.
◦[https://arxiv.org/abs/1611.01646]
Attention Correctness in Neural Image Captioning, Chenxi Liu, Junhua Mao, Fei Sha, Alan Yuille, AAAI 2017
◦[https://arxiv.org/abs/1605.09553]
Text-guided Attention Model for Image Captioning, Jonghwan Mun, Minsu Cho, Bohyung Han, AAAI 2017
◦[https://arxiv.org/abs/1612.03557]
STAIR Captions Constructing a Large-Scale Japanese Image Caption Dataset, Yuya Yoshikawa, Yutaro Shigeto, Akikazu Takeuchi, ACL 2017.
◦[https://arxiv.org/abs/1705.00823]
Actor-Critic Sequence Training for Image Captioning, Li Zhang, Flood Sung, Feng Liu, Tao Xiang, Shaogang Gong, Yongxin Yang, Timothy M. Hospedales, 2017.
◦[https://arxiv.org/abs/1706.09601]
Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering, CVPR2017, Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang.
◦[https://arxiv.org/abs/1707.07998]
Captioning Images with Diverse Objects, CVPR 2017.[http://arxiv.org/abs/1606.07770]
Dense Captioning with Joint Inference and Visual Context, CVPR 2017.[https://arxiv.org/abs/1611.06949]
Incorporating Copying Mechanism in Image Captioning for Learning Novel Objects, CVPR 2017.[https://arxiv.org/abs/1708.05271]
Skeleton Key Image Captioning by Skeleton-Attribute Decomposition, CVPR 2017.[https://arxiv.org/abs/1704.06972]
Towards Diverse and Natural Image Descriptions via a Conditional GAN, ICCV 2017.[https://arxiv.org/abs/1703.06029]
An Empirical Study of Language CNN for Image Captioning, ICCV 2017.[http://arxiv.org/abs/1612.07086]
Areas of Attention for Image Captioning, ICCV 2017.[http://arxiv.org/abs/1612.01033]
Improved Image Captioning via Policy Gradient optimization of SPIDEr, ICCV 2017.[https://arxiv.org/abs/1612.00370]
Paying Attention to Descriptions Generated by Image Captioning Models, ICCV 2017.[http://arxiv.org/abs/1704.07434]
Scene Graph Generation from Objects, Phrases and Region Captions, ICCV 2017[https://arxiv.org/abs/1707.09700]
Show, Adapt and Tell Adversarial Training of Cross-domain Image Captioner, ICCV 2017.[https://arxiv.org/abs/1705.00930]
Speaking the Same Language Matching Machine to Human Captions by Adversarial Training, ICCV 2017.[http://arxiv.org/abs/1703.10476]

中英文综述
Connecting Images and Natural Language, 连接图像与自然语言 Andrej Karpathy, 博士毕业论文, 2016 ak大神 必属精品 保持AK一贯简洁明了的写作风格，非常适合入门学习 摘要：人工智能领域的一个长期目标是开发能够感知和理解我们周围丰富的视觉世界，并能使用自然语言与我们进行关于其的交流的代理。由于近些年来计算基础设施、数据收集和算法的发展，人们在这一目标的实现上已经取得了显著的进步。这些进步在视觉识别上尤为迅速——现在计算机已能以可与人类媲美的表现对图像进行分类，甚至在一些情况下超越人类，比如识别狗的品种。但是，尽管有许多激动人心的进展，但大部分视觉识别方面的进步仍然是在给一张图像分配一个或多个离散的标签（如，人、船、键盘等等）方面。 在这篇学位论文中，我们开发了让我们可以将视觉数据领域和自然语言话语领域连接起来的模型和技术，从而让我们可以实现两个领域中元素的互译。具体来说，首先我们引入了一个可以同时将图像和句子嵌入到一个共有的多模态嵌入空间（multi-modal embedding space）中的模型。然后这个空间让我们可以识别描绘了一个任意句子描述的图像，而且反过来我们还可以找出描述任意图像的句子。其次，我们还开发了一个图像描述模型（image captioning model），该模型可以根据输入其的图像直接生成一个句子描述——该描述并不局限于人工编写的有限选择集合。最后，我们描述了一个可以定位和描述图像中所有显著部分的模型。我们的研究表明这个模型还可以反向使用：以任意描述（如：白色网球鞋）作为输入，然后有效地在一个大型的图像集合中定位其所描述的概念。我们认为这些模型、它们内部所使用的技术以及它们可以带来的交互是实现人工智能之路上的一块垫脚石，而且图像和自然语言之间的连接也能带来许多实用的益处和马上就有价值的应用。 从建模的角度来看，我们的贡献不在于设计和展现了能以复杂的处理流程处理图像和句子的明确算法，而在于卷积和循环神经网络架构的混合设计，这种设计可以在一个单个网络中将视觉数据和自然语言话语连接起来。因此，图像、句子和关联它们的多模态嵌入结构的计算处理会在优化损失函数的过程中自动涌现，该优化考虑网络在图像及其描述的训练数据集上的参数。这种方法享有许多神经网络的优点，其中包括简单的均质计算的使用，这让其易于在硬件上实现并行；以及强大的性能——由于端到端训练（end-to-end training）可以将这个问题表示成单个优化问题，其中该模型的所有组件都具有一个相同的最终目标。我们的研究表明我们的模型在需要图像和自然语言的联合处理的任务中推进了当前最佳的表现，而且我们可以一种能促进对该网络的预测的可解读视觉检查的方式来设计这一架构。
◦[http://cs.stanford.edu/people/karpathy/main.pdf]
Automatic Description Generation from Images: A Survey of Models, Datasets, and Evaluation Measures 2016
◦[https://www.jair.org/media/4900/live-4900-9139-jair.pdf]

# Tutorial
Vision and Language: Bridging Vision and Language with Deep Learning 深度学习为视觉和语言之间搭建了一座桥梁 梅涛博士 微软亚洲研究院资深研究员 2017/09/07。
◦[https://www.microsoft.com/en-us/research/publication/tutorial-bridging-video-language-deep-learning/]
◦[https://pan.baidu.com/s/1o8K5AMu]
Automated Image Captioning with ConvNets and Recurrent Nets Andrej Karpathy关于NeuralTalk一个小时的讲座。
◦[http://cs.stanford.edu/people/karpathy/sfmltalk.pdf]

# 视频教程
cs231 课程第十讲 循环神经网络
◦课件：[https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv]
◦墙外：[https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv]
◦墙内：[http://study.163.com/course/introduction/1003223001.htm]
Automated Image Captioning with ConvNets and Recurrent Nets Karpathy关于NeuralTalk一个小时的视频讲座
◦PPT：[http://cs.stanford.edu/people/karpathy/sfmltalk.pdf]
◦视频：[https://www.youtube.com/watch?v=xKt21ucdBY0]
◦[https://github.com/kelvinxu/arctic-captions]

# 代码
Andrej Karpathy 源于neuraltalk的代码 Torch实现
◦[https://github.com/karpathy/neuraltalk]
◦[https://github.com/karpathy/neuraltalk2]
◦ demo: [http://cs.stanford.edu/people/karpathy/deepimagesent/generationdemo/]
Show and Tell # 代码
◦[https://github.com/kelvinxu/arctic-captions]
Long-term Recurrent Convolutional Networks(LRCN)
◦[https://github.com/jeffdonahue/caffe/tree/54fa90fa1b38af14a6fca32ed8aa5ead38752a09/examples/coco_caption]
Dense image captioning in Torch
◦[https://github.com/jcjohnson/densecap]
TensorFlow (TensorLayer) Implementation of Image Captioning
◦[https://github.com/zsdonghao/Image-Captioning]
Code for paper Generating Images from Captions with Attention by Elman Mansimov, Emilio Parisotto, Jimmy Ba and Ruslan Salakhutdinov; ICLR 2016.
◦[https://github.com/emansim/text2image]
implementation for Self-critical Sequence Training for Image Captioning
◦[https://github.com/ruotianluo/self-critical.pytorch]

# 领域专家
Andrej Karpathy AI届的网红，李飞飞高徒Andrej KarPathy2015年在斯坦福大学获得计算机科学专业博士，2016进入OpenAI工作，2017年6月出任特斯拉AI主管。曾主讲斯坦福著名课CS231N深度学习与计算机视觉。主要研究兴趣为：深度学习，生成模型和强化学习。2011至2015年先后在Google Brain,Deepmind及各大DL实验室实习过，在学习与工作方面经验颇丰。
◦[http://cs.stanford.edu/people/karpathy/]
Jeff Donahue Deepmind 研究科学家，2017年伯克利毕业，贾扬清的师弟，同时也是caffe的主要代码贡献者之一。
◦主页：[http://jeffdonahue.com/]
梅涛 微软亚洲研究院资深研究员
梅涛博士，微软亚洲研究院资深研究员，国际模式识别学会会士，美国计算机协会杰出科学家，中国科技大学和中山大学兼职教授博导。主要研究兴趣为多媒体分析、计算机视觉和机器学习，发表论文 100余篇（h-index 42），先后10次荣获最佳论文奖，拥有40余项美国和国际专利（18项授权），其研究成果十余次被成功转化到微软的产品和服务中。他的研究团队目前致力于视频和图像的深度理解、分析和应用。他同时担任 IEEE 和 ACM 多媒体汇刊（IEEE TMM 和 ACM TOMM）以及模式识别（Pattern Recognition）等学术期刊的编委，并且是多个国际多媒体会议（如 ACM Multimedia, IEEE ICME, IEEE MMSP 等）的大会主席和程序委员会主席。他分别于 2001 年和 2006 年在中国科技大学获学士和博士学位。
◦主页：[https://www.microsoft.com/en-us/research/people/tmei/]
Micah Hodosh 伊利诺伊大学博士
◦[https://scholar.google.com/citations?user=YlaIB0IAAAAJ&hl=en]
Hao Fang 华盛顿大学博士，Microsoft COCO 数据集的作者之一
◦[http://students.washington.edu/hfang/]
Junhua Mao (毛俊骅) 加州大学洛杉矶分校博士，本科中科大，曾获科大最高奖郭沫若奖学金。在百度研究院实习期间最早提出了生成方式的image caption工作。其后分别在Google Research，Pinterest以及Google X 自动驾驶项目实习。
◦[http://www.stat.ucla.edu/~junhua.mao/]
Subhashini Venugopalan UT 奥斯丁大学博士，曾在Google Research实习。
◦[https://vsubhashini.github.io/]
Xinlei Chen 陈鑫磊 CMU博士，专注文本与图像交叉领域。
◦[https://www.cs.cmu.edu/~xinleic/]
Larry Zitnick Facebook人工智能研究院（FAIR）主要研究员，华盛顿大学副教授。
◦[http://larryzitnick.org/]
Quanzeng You Rochester大学博士生，导师是下面的Jiebo Luo
◦[http://www.cs.rochester.edu/u/qyou/]
Jiebo Luo IEEE/SPIE Fellow、长江讲座美国罗彻斯特大学教授 ◦[http://www.cs.rochester.edu/u/jluo/12]. Justin Johnson 和Andrej Karpathy一样，都是李飞飞高徒。cs231n课程主讲之一
◦[http://cs.stanford.edu/people/jcjohns/]
Samy Bengio Google研究科学家，大神Yoshua Bengio的弟弟。关注多模态研究。
◦[http://bengio.abracadoudou.com/]
Qi Wu (吴琦) 阿德莱德大学Research Fellow
◦[http://qi-wu.me/]

# 数据集
Microsoft COCO Caption数据集 Microsoft COCO Caption数据集的推出，是建立在Microsoft Common Objects in COntext (COCO)数据集的工作基础上的。在论文《Microsoft COCO Captions: Data Collection and Evaluation Server》中，作者们详细介绍了他们基于MS COCO数据集构建MS COCO Caption数据集的工作。可以在这里看到最近的结果排名。
◦leaderboard：[https://competitions.codalab.org/competitions/3221#results]
Flickr8K和30K Flickr8K和Flickr30K数据集的图像数据来源是雅虎的相册网站Flickr，数据集中图像的数量分别是8,000张和31,783张；这两个数据库中的图像大多展示的是人类在参与到某项活动中的情景。每张图像的对应人工标注依旧是5句话。这两个数据库本是同根生，所以其标注的语法比较类似。数据库也是按照标准的训练集、验证集合测试集来进行分块的。相较于MS COCO Caption数据集，Flickr8K和Flickr30K数据集的明显劣势就在于其数据量不足。
◦[http://shannon.cs.illinois.edu/DenotationGraph/]
◦[http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html]
PASCAL 1K 该数据集的图像是大名鼎鼎的PASCAL VOC challenge图像数据集的一个子集，对于其20个分类，随机选出了50张图像，共1,000张图像。然后同样适用亚马逊公司的土耳其机器人服务为每张图像人工标注了5个描述语句。一般说来，这个数据集只是用来测试的。
◦[http://nlp.cs.illinois.edu/HockenmaierGroup/pascal-sentences/index.html]
# 参考
1. 图像描述生成（Image Caption）. http://www.zhuanzhi.ai/topic/2001990892850700/awesome