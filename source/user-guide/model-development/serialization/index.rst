.. _serialization-guide:

=====================
保存与加载模型（S&L）
=====================

在模型开发的过程中，我们经常会遇到需要保存（Save）和加载（Load）模型的情况，例如：

* 为了避免不可抗力导致的训练中断，需要养成模型每训练一定时期（Epoch）就进行保存的好习惯；
* 同时如果训练时间过长，可能会导致模型在训练数据集上过拟合，因此需要保存多个检查点，取最优结果；
* 某些情况下，我们需要加载预训练模型的参数和其它必需信息，恢复训练或进行微调...

在 MegEngine 中对 Python 自带的 :mod:`pickle` 模块进行了封装，
来实现对 Python 对象结构（如 Module 对象）的二进制序列化和反序列化。
其中需要被我们熟知的核心接口为 :func:`megengine.save` 和 :func:`megengine.load`:

>>> megengine.save(model, PATH)
>>> model = megengine.load(PATH)

上述语法非常简明直观地对整个 ``model`` 模型进行了保存和加载，但这并不是推荐做法。
更加推荐的做法是保存和加载 ``state_dict`` 对象，或使用检查点（Checkpoint）技术。
接下来将对上面的内容做更加具体的解释，并提供一些情景下保存和加载模型的最佳实践。
你可以略过已经熟悉的概念，直接跳转到所需的用例代码展示。

.. list-table:: 

   * - :ref:`save-load-entire-model`
     - 任何情况下都不推荐 ❌
   * - :ref:`save-load-model-state-dict`
     - 适用于推理 ✅ 不满足恢复训练要求 😅 
   * - :ref:`save-load-checkpoint`
     - 适用于推理或恢复训练 💡
   * - :ref:`dump-traced-model` （Dump）
     - 适用于推理，且追求高性能部署 🚀

.. note::

   使用 ``pickle`` 模块时，相应术语也叫做封存（pickling）和解封（unpickling）。

.. admonition:: pickle 模块与协议的兼容
   :class: note

   由于不同版本的 Python 之间 ``pickle`` 模块使用的 
   `数据流格式 <https://docs.python.org/3/library/pickle.html#data-stream-format>`_ 协议可能不同，
   因此可能会出现高版本 Python 保存的 MegEngine 模型在低版本 Python
   无法加载的情况。这里提供两种解决思路：

   * 在调用 :func:`megengine.save` 时，通过参数 ``pickle_protocol`` 指定兼容性较强的版本（比如第 4 版）;
   * 接口 :func:`megengine.save` 和 :func:`megengine.load` 都支持传入 ``pickle_module`` 参数，
     从而使用指定的 ``pickle`` 模块，比如安装并使用 `pickle5 <https://pypi.org/project/pickle5/>`_ 
     来代替 Python 内置的 ``pickle`` 模块：

     >>> import pickle5 as pickle

.. admonition:: pickle 模块并不安全！
   :class: warning

   * 不坏好意的人可以通过构建恶意的 ``pickle`` 数据来在解封时执行任意代码；
   * 因此绝对不要对不信任来源的数据和可能被篡改过的数据进行解封。


下面是我们用于举例的 ``ConvNet`` 模型：

.. code-block:: python

   import megengine.functional as F
   import megengine.module as M
   import megengine.optimizer as optim

   class ConvNet(M.Module):
      def __init__(self):
         super().__init__()
         self.conv1 = M.Conv2d(1, 10, 5)
         self.pool1 = M.MaxPool2d(2, 2)
         self.conv2 = M.Conv2d(10, 20, 5)
         self.pool2 = M.MaxPool2d(2, 2)
         self.fc1 = M.Linear(320, 50)
         self.fc2 = M.Linear(50, 10)

      def forward(self, input):
         x = self.pool1(F.relu(self.conv1(input)))
         x = self.pool2(F.relu(self.conv2(x)))
         x = F.flatten(x, 1)
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))

         return x

   model = ConvNet()

   optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

.. _save-load-entire-model:

保存/加载整个模型
-----------------

保存：

>>> megengine.save(model, PATH)

加载：

>>> model = megengine.load(PATH)
>>> model.eval()

.. note::

   我们不推荐使用这种方法的原因在于 ``pickle`` 本身的局限性：对于特定的类，如用户自己设计的一个 ``ConvNet`` 模型类，
   ``pickle`` 在保存该模型时不会序列化模型类本身，而是会将该类与包含其定义的源码的路径绑定，如 ``project/model.py``.
   在加载模型时， ``pickle`` 需要用到此路径。因此如果在后续的开发过程中，你对项目进行了重构
   （比如将 ``model.py`` 进行了重命名），将导致执行模型加载的步骤时失败。

.. warning::

   如果你依旧使用这种方法加载模型并尝试进行推理，记得先调用 ``model.eval()`` 切换到评估模式。

.. _save-load-model-state-dict:

保存/加载模型状态字典
---------------------

保存：

>>> megengine.save(model.state_dict(), PATH)

加载：

>>> model = ConvNet()
>>> model.load_state_dict(megengine.load(PATH))
>>> model.eval()

当保存一个仅用作推理的模型时，必须进行的处理是保存模型中学得的参数（Learned parameters）。
相较于保存整个模型，更加推荐保存模型的状态字典 ``state_dict``, 在后续恢复模型时将更加灵活。

.. warning::

   * 相较于加载整个模型的做法，此时 ``megengine.load()`` 得到的结果是一个状态字典对象，
     因此还需要通过 ``model.load_state_dict()`` 方法进一步将状态字典加载到模型中，
     不能够使用 ``model = megengine.load(PATH)``; 另一种常见的错误用法是直接 ``model.load_state_dict(PATH)``,
     注意必须先通过 ``megengine.load()`` 反序列化得到状态字典，再传递给 ``model.load_state_dict()`` 方法；
   * 加载状态字典成功后，记得调用 ``model.eval()`` 将模型切换到评估模式。

.. note::

   通常我们约定使用 ``.pkl`` 文件扩展名保存模型，如 ``mge_checkpoint_xxx.pkl`` 形式。

.. admonition:: 注意 ``.pkl`` 与 ``.mge`` 文件的区别
   :class: warning

   ``.mge`` 文件通常是 MegEngine 模型经过 :ref:`dump` 得到的文件，用于推理部署。

什么是状态字典
~~~~~~~~~~~~~~

由于使用 ``pickle`` 直接 :ref:`save-load-entire-model` 时存在受到路径影响的局限性，
我们则需要考虑使用原生的 Python 数据结构来记录模型内部的状态信息，方便进行序列化和反序列化。
在 :ref:`module-design` 中，我们提到了每个 Module 有一个状态字典成员，
记录着模型内部的 Tensor 信息（即 :ref:`parameter-and-buffer` ）：

>>> for tensor in model.state_dict():
...     print(tensor, "\t", model.state_dict()[tensor].shape)
conv1.bias 	 (1, 10, 1, 1)
conv1.weight 	 (10, 1, 5, 5)
conv2.bias 	 (1, 20, 1, 1)
conv2.weight 	 (20, 10, 5, 5)
fc1.bias 	 (50,)
fc1.weight 	 (50, 320)
fc2.bias 	 (10,)
fc2.weight 	 (10, 50)

状态字典是一个简单的 Python 字典对象，因此可以借助 ``pickle`` 轻松地保存和加载。


.. note::

   每个优化器 ``Optimzer`` 也有一个状态字典，其中包含有关优化器状态的信息，以及使用的超参数；
   如果后续有恢复模型并且继续训练的需求，仅保存模型的状态字典是不行的 ——
   我们同时还需要保存优化器的状态字典等信息，即下面提到的 “检查点” 技术。

.. seealso::

   关于状态字典的进一步解释： :ref:`module-state-dict` / :ref:`optimizer-state-dict`

.. _save-load-checkpoint:

保存/加载检查点
---------------

保存：

.. code-block:: python

   megengine.save({
                   "epoch": epoch,
                   "state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "loss": loss,
                   ...
                  }, PATH)
   
加载：

.. code-block:: python

   model = ConvNet()
   optimizer = optim.SGD()

   checkpoint = megengine.load(PATH)
   model.load_state_dict(checkpoint["model_state_dict"])
   optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
   epoch = checkpoint["epoch"]
   loss = checkpoint["loss"]

   model.eval()
   # - or -
   model.train()

* 保存检查点是为了能够恢复到和训练时一致的状态：
  需要恢复的不仅仅是 :ref:`module-state-dict` ，:ref:`optimizer-state-dict`.
  根据实际需求，还可以记录训练时达到的 ``epoch`` 以及最新的 ``loss`` 信息。
* 加载检查点后，根据是希望继续训练，还是用作推理来设置模型为训练或评估模式。

.. warning::

   相较于仅保存模型的状态字典，保存完整检查点会占据比较多的硬盘空间。
   因此如果你十分确定以后只需要进行模型推理时，可以不必保存检查点。
   亦或者设定不同的保存频率，例如每 10 个 Epochs 保存一次状态字典，
   每 100 个 Epochs 保存一次完整的检查点，这取决于你的实际需求。

.. seealso::

   参考官方 ResNet 模型中如何保存和加载检查点：

   :models:`official/vision/classification/resnet`

   在 ``train/test/inference.py`` 可找到相关接口。

.. _dump-traced-model:

导出静态图模型
--------------

为了将最终训练好的模型部署到生产环境，模型开发的最后一步需要导出静态图模型：

.. code-block:: python

   from megengine import jit

   model = ConvNet()
   model.load_state_dict(megengine.load(PATH))
   model.eval()

   @jit.trace(symbolic=True, capture_as_const=True)
   def infer_func(data, *, model):
       pred = model(data)
       pred_normalized = F.softmax(pred)
       return pred_normalized

   data = megengine.Tensor(np.random.randn(1, 1, 28, 28))
   output = infer_func(data, model=model)
   
   infer_func.dump(PATH, arg_names=["data"])

.. seealso::

   更加具体的解释请参考： :ref:`dump` 。


