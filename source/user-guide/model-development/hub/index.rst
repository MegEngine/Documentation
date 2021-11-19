.. _hub-guide:

=============================
使用 Hub 发布和加载预训练模型
=============================

借助于 MegEngine 提供的 :mod:`~.hub` 模块，研究人员能够很方便地：

* 通过添加一个 ``hubconf.py`` 文件，对自己开源在 GitHub 或 GitLab 上的预训练模型进行发布；
* 通过 :func:`megengine.hub.load` 接口加载其他研究人员发布的预训练模型，有利于进行研究结果的复现；
* 加载完成的预训练模型还可以用作迁移学习的微调，或是进行预测。

这一小节将以 `ResNet <https://github.com/MegEngine/Models/tree/master/official/vision/classification/resnet>`_ 
系列模型为例子，展示模型的发布和加载流程。

.. note::

   此处的 “预训练模型” 包括 1. 模型的定义 2. 预训练权重。

   相较于使用 :meth:`.Module.load_state_dict` 和 :func:`megengine.load` 反序列化并加载模型的状态字典，
   :func:`megengine.hub.load` 还能够在此之前替用户完成加载模型定义的过程（根据 ``hubconf.py`` ），
   加载后的模型也可被用于 :ref:`dump` ，进而用于高性能的推理部署情景。

.. note::

   Hub 的相关功能也可以用作内部 Git 服务器，需在使用相关接口时配置相应的参数。

.. seealso::

   MegEngine 官网上提供了 `模型中心 <https://megengine.org.cn/model-hub>`_ 板块，
   基于旷视研究院领先的深度学习算法，提供满足多业务场景的预训练模型。
   实际上即对官方模型库 Models 添加了 :models:`hubconf.py` 配置。
   如果你希望将自己的研究模型发布到 MegEngine 官方的模型中心，请参考
   `Hub <https://github.com/MegEngine/Hub>`_ 存储库的 README 文件。

发布预训练模型
--------------

在 ``hubconf.py`` 文件中，需要将提供至少一个入口点（Entrypoint），形式如下：

.. code-block:: python

   def entrypoint_name(*args, **kwargs):
       """Returns a model."""
       ...

* 调用入口点时，通常返回一个模型 ( M.Module ), 也可以是其他希望通过 Hub 加载的对象；
* 加载模型时 ``*args`` 和 ``**kwargs`` 参数将被传递给真正的可调用对象；
* 入口点的文档字符串会在调用 :func:`.hub.help` 接口时显示。

提供入口点
~~~~~~~~~~

以官方 `ResNet <https://github.com/MegEngine/Models/tree/master/official/vision/classification/resnet>`_ 模型为例，
模型定义文件在 :models:`official/vision/classification/resnet/model.py`.

.. code-block:: shell

    Models
    ├── official/vision/classification/resnet
    │   └── model.py
    └── hunconf.py

我们在 ``hubconf.py`` 中可以这样实现一个入口点：

.. code-block:: python

   from official.vision.classification.resnet.model import BasicBlock, Bottleneck, ResNet

   def resnet18(**kwargs):
       """Resnet18 model"""
       return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

不过由于 ResNet 的 ``model.py`` 中已经对 
``resnet18``, ``resnet34``, ``resnet50``, ``resnet101``, ``resnet152``
等常见的网络结构按照 Hub 风格进行了定义，因此实际的 :models:`hubconf.py` 中只需要进行导入它们即可：

.. code-block::

   from official.vision.classification.resnet.model import (
        ...
        resnet18,
        ...
    )

提供预训练权重
~~~~~~~~~~~~~~

通过给入口点添加 :class:`.hub.pretrained` 装饰器，来标识预训练权重的 URL 地址：

.. code-block:: python

   @hub.pretrained("https://url/to/pretrained_resnet18.pkl")
   def resnet18(pretrained=False, **kwargs):
       """Resnet18 model"""
       return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

* 当被装饰的函数具有参数 ``pretrained=True`` 时，调用时将自动下载并对返回的模型填入预训练的权重；
* 预训练权重可以存在于 Git 存储库中，对于开源在 GitHub / GitLab 的项目，
  需要考虑预训练权重的体积整体大小以及用户的下载条件，可根据实际情况判断 ——
  选择将预训练权重附加和模型放在一起发布，还是放在其它位置（比如网盘、OSS 等）。

加载预训练模型
--------------

通过 :func:`.hub.list` 接口可以列举出指定的 GitHub 库中 ``hubconf.py`` 中提供的预训练模型入口。

例如运行下面这个命令，可以找到 GitHub 的 MegEngine/Models 库中所有发布的预训练模型：

>>> megengine.hub.list("megengine/models")
['ATSS',
 'BasicBlock',
 # ...
 'resnet18',
 # ...
 'wwm_cased_L_24_H_1024_A_16',
 'wwm_uncased_L_24_H_1024_A_16']

假定我们需要的是 ``resnet18`` 预训练模型，使用 :func:`.hub.help` 接口，可以查看对应入口点的文档字符串信息：

>>> megengine.hub.help("megengine/models", "resnet18")
'ResNet-18 model...'

只需要使用 :func:`.hub.load` 接口，便可以一次性地完成对应预训练模型的加载：

>>> model = megengine.hub.load('megengine/models', 'resnet18', pretrained=True)
>>> model.eval()

.. warning::

   在推理之前，记得调用 ``model.eval()`` 将模型切换到评估模式。

.. note::

   默认情况下， 将从 GitHub 对应存储库的 master 分支进行 ``hubconf.py`` 等文件的拉取：

   * 可通过形如 ``megengine/models:dev`` 的形式指定到 dev 分支名（或标签名）；
   * 可通过设置 ``git_host`` 参数，选择使用指定的 Git 服务器；
   * 可通过设置 ``commit`` 参数，选择使用指定的 commit 位置；
   * 可通过设置 ``protocol`` 参数，选择获取代码仓库时所用的协议。

   通常情况下，无需进行额外设置，可以从公开的 GitHub 库以 HTTPS 协议进行 Clone.
   如果进行了具体配置（如使用了内部 Git 服务器），请确保你有对应代码仓库的访问权限。

