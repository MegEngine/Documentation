.. _model-development-basic:

======================
MegEngine 虚拟炼丹挑战
======================

.. warning:: 本页面内容还在撰写中...

.. admonition:: 本教程涉及的内容
   :class: note

   * 改进上一个教程中的卷积神经网络模型，理解 MegEngine/Models 库中的 ResNet 官方实现；
   * 学习如何保存和加载模型，以及如何使用预训练模型进行微调（迁移学习）；
   * 整合模型开发常见技巧，总结本系列教程，提供接下来可能的几个学习方向作为参考。

.. warning::

   * 本教程的文风比较独特，假定你正用 MegEngine 参与一场竞赛，希望你能获得有如身临其境的感觉；
   * 我们最终并不会真正地去从零训练一个 ResNet 模型（用时太久），而是掌握其思想。

.. seealso::

   示范代码大都来自： :models:`official/vision/classification/resnet` （我们的目的是读懂它）

ImageNet 数据集
---------------

回忆一下在前面几个教程中，我们的讲解思路都是大同小异的，有大量的重复代码模式：

1. 通常我们会介绍一个简单的模型并实现它，然后用于当前教程的数据集；
2. 在下一个教程中，我们仅仅换用了一个更加复杂的数据集，便发现了模型效果不佳；
3. 因此需要设计出更好的模型（借机引入新的相关知识），并进行训练和验证。

在第 2 步中，我们需要修改的主要是数据加载部分，比如调用不同的数据集获取接口；
而在第 3 步中，数据加载则不再被关心，我们专注于模型的设计和超参数的调整。
实际上对于同样类型的任务，相同的模型设计可能可以用于不同的数据集，
甚至只需要在上一个训练好的模型中做一些微调，就能在类似的任务中取得不错的效果。
这不禁让人思考 —— 如何去优化模型开发的流程，以便在面对不同的任务时能够做到更加高效？

这个教程还将帮助你思考一些模块化与工程化问题，接触实际生产中可能采用的工作流程。

了解数据集信息
~~~~~~~~~~~~~~

.. dropdown:: :fa:`eye,mr-1` ImageNet 数据集与 ILSVRC

   `ImageNet <https://www.image-net.org/>`_ :footcite:p:`deng2009imagenet`
   里面具有 14,197,122  张图片，21841 个同义词集索引。
   该数据集的子集被用于 ImageNet 大规模视觉识别挑战赛 (ILSVRC)，作为图像分类和目标检测的基准。
   “ILSVRC” 一词有时候也用来特指该比赛使用的数据集（即刚才提到的 ImageNet 的一个子集），
   其中最常用的是 2012 年的数据集，记为 ILSVRC2012.
   因此有时候提到 ImageNet, 很可能是指 ImageNet 中用于 ILSVRC2012 的这个子集。

   MegEngine/Models 中的 ResNet 默认在 ILSVRC2012  数据集上进行训练和评估，
   ILSVRC2012 中拥有越 120 万张图片作为训练集，5 万张图片作为验证集，
   10 万张图片作为测试集（不带标记），总共有 1000 个分类。
   ILSVRC2012 有时也叫 ILSVRC 1K, 1K 指的是 1000 个类别。

   我们曾在前面的教程中提到过数据集划分的概念，ILSVRC 就是很好的一个例子，
   比赛方会提供给你带有标记的训练集、验证集，但最终实际用来排名的是测试集。
   参赛人员没有任何途径获取测试集对应的标记，只能将自己的模型在测试集上的预测结果提交给比赛平台，
   由后台机器评委结合测试集的标记对最终的结果进行评估（评估指标是公开的）。
   在科研文献中，通常比较的是在训练集和验证集上的评估结果。

   ResNet 模型获得了2015 年 ILSVRC 的冠军，且其设计思路经过了时间的验证，可谓经典。

.. warning::

   使用频率最高的 ILSVRC2012 图像分类和定位数据集目前可以在
   `Kaggle <https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description>`_
   获取。但完整的 ImageNet 和其它常用子集并不可以直接地进行获取，
   虽然 MegEngine 中提供了相应的 :class:`~.ImageNet` 处理接口，
   但也仅仅用于对本地数据进行处理，不会进行原始数据集的下载。

.. admonition:: ImageNet 的评估指标

   * Top-1 准确率：预测分类概率中，概率最大的类别与真实的标记一致，即认为正确；
   * Top-5 准确率：预测分类概率中，概率前五的类别中包含真实标记类别，即认为正确。

   这对应了 MegEngine 中的 :func:`~.topk_accuracy` 接口，如果用错误率表示，则为 1 - 正确率。

参加 Virtual ILSVRC
-------------------

将时间拨回到平行宇宙的 2015 年，你作为 MegEngine 炼丹炉代表队的一员主力，
刚学会了卷积神经网络，原本尝试用 LeNet5 模型参与 ILSVRC 比赛，结果自不用猜... 完全没法看。
铩羽而归绝不是小队的目标，划水观光更不能接受，
现在需要想方设法地进行改进，争取在今年的挑战赛上打出风采。

恺铭、祥禹、韶卿决定加入你的队伍，经验丰富的孙健老师将作为指导，一齐参与 ILSVRC2015 挑战赛！

.. dropdown:: :fa:`eye,mr-1` 真实情况是...

   何恺明、张祥雨、任少卿和孙剑是论文《Deep Residual Learning for Image Recognition》的作者，
   没错这就是 ResNet 模型的对应论文，它获得了CVPR 2016 Best Paper.

   注：CVPR 是国际计算机视觉与模式识别会议（Conference on Computer Vision and Pattern Recognition）
   的缩写。作为 IEEE 一年一度的学术性会议，会议的主要内容是计算机视觉与模式识别技术。
   CVPR 是世界顶级的计算机视觉会议，你可以尝试使用 MegEngine 复现很多经典论文中的实验结果。

   后文的情节与实际的历史会有比较大的差异（虚构），会通过此类形式进行说明。

.. figure:: ../../_static/images/ILSVRC.jpg

那么问题来了，要如何去做改进呢？解决问题的思路很重要，大家决定从不同的角度来想想办法。

孙老师说：“让我们先来看看过去几年的 ILSVRC 图像分类冠亚军能提供些什么思路吧。”

相关的论文祥禹早已烂熟于心，很快他给出了几篇需要被重点关注的对象：AlexNet, VGGNet, GoogleNet...
“这几篇论文的处理思路、模型结构都挺新颖的，值得一看。” 于是大家决定按照时间顺序，从 AlexNet 开始看起。

加大炼丹火力
------------

传统神经网络中使用 :func:`~.nn.sigmoid` 或 :func:`~.tanh` 作为激活函数，
AlexNet 中使用了 :func:`~.nn.relu`, 这个做法你们已经应用。另外你还注意到，
AlexNet 使用了 2 个 GPU 进行训练！ **“我们需要更多的 GPU 来节省时间！”**  你激动地喊道。

.. dropdown:: :fa:`eye,mr-1` 真实情况是...

   使用多个 GPU 设备涉及到 :ref:`distributed-guide` 的概念，相较于单卡训练，这确实能够节省时间。
   但在当时的历史背景下，作者 `Alex Krizhevsky <https://www.cs.toronto.edu/~kriz/>`_
   使用两个 GPU 的实际原因是当时所用的 GPU 设备（GTX 580）内存不足以存储下 AlexNet 中的所有参数，
   因此画出来的模型结构是这样的：

   .. figure:: ../../_static/images/alexnet_paper.png

      来自论文 `ImageNet Classification with Deep Convolutional Neural Networks
      <https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf>`_

   如今的 GPU 设备内存容量足以放下完整的 AlexNet 结构，大部分单卡 GPU 即可进行复现。

.. seealso::

   在 :models:`official/vision/classification/resnet/train.py#L112` 中支持单个或多个
   GPU 进行 ResNet 的训练，其中每台 GPU 设备被看作是一个 ``worker``.
   多个 GPU 设备训练时需要关注各种数据同步策略，例如：

   .. code-block:: python

      # Sync parameters and buffers
      if dist.get_world_size() > 1:
          dist.bcast_list_(model.parameters())
          dist.bcast_list_(model.buffers())

   从单卡到多卡，需要用到 :class:`~.distributed.launcher` 装饰器，更多介绍请参考 :ref:`distributed-guide` 。

只见孙老师大手一挥：“没问题，给你八张体质贼棒、性能贼强的卡，咱们把火力拉满。”

提升灵材品质
------------

你正沉迷在多卡妙用的奇思妙想之中，这时候韶卿提醒大家：“AlexNet 还做了数据增强，咱们也可以试试。”

.. _data-augmentation:

数据增强
~~~~~~~~

俗话说 “见多识广”，越大的数据集通常可以带来越好的模型性能，
因此数据增强（Data augmentation）是一种十分常见的预处理手段。
但 ImageNet 比赛不允许使用其它的数据集，因此能够采取的做法便是对原有数据集中的图片进行一些随机的处理，
比如随机平移、翻转等等。对于计算机来说，这样的图片可以被看做是不同的，
随机因素使得每次得到的分批数据也都是不同的。举例效果如下：

.. figure:: ../../_static/images/chai-data-augmentation.png/

MegEngine 的 :mod:`.data.transform` 模块中对常见的图片数据变换都进行了实现，可以在加载数据时进行：

.. code-block:: python
   :emphasize-lines: 6-7

   train_dataloader = data.DataLoader(
       train_dataset,
       sampler=train_sampler,
       transform=T.Compose(
           [  # Baseline Augmentation for small models
              T.RandomResizedCrop(224),
              T.RandomHorizontalFlip(),
              T.Normalize(
                   mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
              ),  # BGR
              T.ToMode("CHW"),
           ]
       )
   )

“好，这样就可以在加载数据时随机裁剪到 224 的长和宽，并且随机做水平翻转了。”
韶卿快速查了查 MegEngine 的 API 文档，稳稳地将这些操作加上。
同时他也在做 :class:`~.transform.Normalize` 归一化的同时，
标记上了图片的通道顺序，“好习惯呀，这波属实是学到了”，你默默在心里竖起了一个大拇指。

.. dropdown:: :fa:`eye,mr-1` 真实情况是...

   AlexNet 中使用的数据增强操作与这里有些不同，对应 :class:`.transform.Lighting` 接口。

   这里演示的数据增强方式是利用 MegEngine 的接口在加载数据后即时地变换处理，
   也叫做在线增强。一些情景下我们也可以使用对数据离线增强，即提前地用类似 OpenCV 这样的软件做好增强处理，
   这样在加载数据时可以看作是使用了好几个数据集。这种方式需要占用掉更多的空间，
   而在线增强每次仅会对当前 Batch 的数据进行随机处理，用完就不再需要了。

   将验证集（甚至是测试集，如果你能得到）的数据加入训练集中不能够算作是数据增强，
   反而是数据泄露（Data leakage），你的模型可能会在这些数据集上过拟合。

   顺便提一下，AlexNet 中使用到了 :class:`~.module.Dropout` 来防止过拟合，
   我们在 ResNet 模型中不会使用这个技巧。

数据清洗
~~~~~~~~

除了数据增强，你还想到一种可能性：“会不会 ImageNet 本身的数据集质量有问题呢？”

数据的内容和标注质量将对模型的效果造成无法忽视的影响，由于 ImageNet 本质上是一个网络图片数据集，
因此其中会有大量的脏数据 —— 图片内容质量不好，格式不一致（灰度图和彩图混合），标记错误等等情况都存在。
你化身为数据清洗小能手，尝试去人工地清洗这些脏数据，但这样做的效率太低了，于是偏很快放弃。

.. seealso::

   事实上，现在已经有许多不错的数据清洗工具可以帮助我们完成类似的工作，
   比如 `cleanlab <https://github.com/cleanlab/cleanlab>`_
   可以帮助我们找出数据集中错误的标记，
   其官方声明在 ImageNet 数据集中找到了接近 100,000 个错误标记！

   .. figure:: ../../_static/images/imagenet_train_label_errors.jpg

      Top label issues in the 2012 ILSVRC ImageNet train set identified using cleanlab.
      Label Errors are boxed in red. Ontological issues in green. Multi-label images in blue.

   在工业界，你会在一些机器学习团队中看到有专门的数据团队，负责提高数据集质量。合作万岁！

改进控火技术
------------

.. panels::
   :container: +full-width text-center
   :card:

   .. figure:: ../../_static/images/loss-surface-optimization.gif
   ---
   .. figure:: ../../_static/images/sgd-saddle-point.gif

.. dropdown:: :fa:`eye,mr-1` 真实情况是...

   《`An overview of gradient descent optimization algorithms
   <https://arxiv.org/abs/1609.04747>`_》是一篇挂在 Arxiv 上的文章，
   原文形式本是一篇 `博客 <https://ruder.io/optimizing-gradient-descent/index.html>`_ ，
   发布时间是 2016 年，因此这里读论文的情节是虚构的。
   但是其中的一些优化方法如 Momentum 确实在 2013~2015 年就被陆续提出并不断改进。
   MegEngine 中的 :mod:`~.optimizer` 模块是高度可灵活拓展的。


秘制高阶丹方
------------


拓展材料
--------


参考文献
--------

.. footbibliography::



