.. _quantization-guide:

====================
量化（Quantization）
====================
.. toctree::
   :hidden:

   basic-concept

.. note::

   常见神经网络模型所用的 :ref:`tensor-dtype` 一般是 ``float32`` 类型，
   而工业界出于对特定场景的需求，需要把模型转换为像 ``int8`` 这样的低精度/比特类型
   —— 整个过程被称为量化（Quantization）。

   .. mermaid::
      :align: center
      :caption: 通常以浮点模型为起点，经过中间的量化处理后最终变成量化模型

      flowchart LR
          FM[Float Model] -- processing --> QM[Quantized Model]

   * 量化能将 32 位的浮点数转换成 8 位甚至是 4 位定点数，具有更少的运行时内存和缓存要求；
     另外由于大部分的硬件对于定点运算都有特定的优化，所以在运行速度上也会有较大的提升。
     相较于普通模型， **量化模型有着更小的内存容量与带宽占用、更低的功耗和更快的推理速度等优点。**
   * 某些计算设备只支持做定点运算。为了让模型可以在这些设备上正常运行，我们需要进行量化处理。

   “为了追求极致的推理计算速度，从而舍弃了数值表示的精度”，直觉上会带来较大的模型掉点，
   但是在通过一系列精妙的量化处理之后，其掉点可以变得微乎其微，并能支持正常的部署使用。

   用户无需了解背后的实现细节，使用 MegEngine 的 :mod:`~.quantization` 所提供的解决方案，就能满足基本量化需求。
   我们为感兴趣的用户提供了更多有关量化基本原理的介绍，可参考 :ref:`quantization-basic-concept` 。

   熟悉基本原理的用户可直接跳转到 :ref:`megengine-quantization` ↩ 查看基本用法。

.. warning::

   请不要将 “量化” 与 “混合精度（Mixed precision）” 混淆，可参考 :ref:`amp-guide` 文档。
   
.. _quantization-intro:

量化基本流程介绍
----------------

目前工业界主要应用有两类量化技术，在 MegEngine 中都进行了支持：

* 训练后量化（Post-Training Quantization, PTQ）；
* 量化感知训练（Quantization-Aware Training, QAT）。

训练后量化是将已经训练好的浮点模型转换成低精度/比特模型所使用的通用技术，常见的做法是对模型的权重（weight）和激活值（activation）进行处理，
将它们转换成精度更低的类型。在转换过程中，需要用到待量化模型中权重和激活的一些统计信息，如缩放因子（scale）和零点（zero_point）。
尽管精度转换发生在训练后，但为了获取这些统计信息，我们仍需要在模型训练时 —— 即前向计算的过程中，插入一名观察者（Observer）。

使用训练后量化技术，会导致量化后的模型掉点（即预测正确率下降）。严重情况下会导致量化模型不可用。
一种可行的做法是使用小批量数据来在量化前对 Observer 进行校准（Calibration），也叫 Calibration 后量化。

另一种可行的改善方案是使用量化感知训练技术，向浮点模型中插入一些伪量化（FakeQuantize）算子作为改造，
在训练时伪量化算子会根据 Observer 观察到的信息进行量化模拟，
即模拟计算过程中数值截断后精度降低的情形，先做一遍数值转换，再将转换后的值还原成原类型。
这样可以让被量化对象在训练时 “提前适应” 量化操作，缓解在训练后量化时带来的掉点影响。

新增的 FakeQuantize 算子会引入大量的训练开销，为了节省总用时，模型量化更通常的思路是：

#. 按照平时训练模型的流程，设计好 Float 模型并进行训练（等同于得到一个预训练模型）；
#. 插入 Observer 和 FakeQuantize 算子，得到 Quantized-Float 模型（简称 QFloat 模型），量化感知训练；
#. 进行训练后量化，得到真正的 Quantized 模型（简称 Q 模型），即最终被用作推理的低比特模型。

.. mermaid::
   :align: center
   :caption: 此时的量化感知训练 QAT 可被看作是在预训练好的 QFloat 模型上微调（Fine-tune），同时做了校准

   flowchart LR
       FM[Float Model] --> |train| PFM[Pre-trained Float Model]
       PFM --> |Observer| PQFM[Pre-trained QFloat Model]
       PFM --> |FakeQuantize| PQFM
       PQFM --> |QAT| FQFM[Fine-tuned QFloat Model]
       FQFM --> |PTQ| QM[Q Model]

.. note::

   根据实际情景的一些差异，量化流程可以有灵活的变化，例如：

   * 在不考虑训练开销的情况下，为了简化整体流程，可以直接构造 QFloat 模型，并进行训练与后量化：

      .. mermaid::
         :align: center

         flowchart LR
            FM[Float Model] --> |Observer| QFM[QFloat Model]
            FM[Float Model] --> |FakeQuantize| QFM[QFloat Model]
            QFM --> |QAT| TQFM[trained QFloat Model]
            TQFM --> |PTQ| QM[Q Model]

   * 在构造 QFloat 模型时，如果不插入 FakeQuantize 算子，也可以相应地减少训练开销，提升速度。

     但是这时等同于未进行量化感知，只进行了数据校准 Calibration, 模型可能会掉点严重：

      .. mermaid::
         :align: center

         flowchart LR
            PFM[Pre-trained Float Model] --> |Observer| PQFM[Pre-trained QFloat Model]
            PFM[Pre-trained Float Model] -.- |FakeQuantize| PQFM[Pre-trained QFloat Model]
            PQFM --> |Calibration| CQFM[Calibrated QFloat Model]
            CQFM --> |PTQ| QM[Q Model]

   对于上述不同情景，在 MegEngine 中可以使用一套统一的接口来对不同的情况进行灵活配置。

.. _megengine-quantization:

Megengine 量化步骤
------------------

在 MegEngine 中，最上层的量化接口是配置如何量化的 :class:`~.quantization.QConfig` 
和模型转换模块里的 :func:`~.quantization.quantize_qat` 与 :func:`~.quantization.quantize` .
通过配置 :class:`~.quantization.QConfig` 中所使用的 Observer 和 FakeQuantize 算子，我们可以对量化方案进行自定义。
进一步的说明请参考 :ref:`qconfig-guide` 小节，下面将展示 QAT 量化流程所需的步骤:

.. code-block:: python

   import megengine.quantization as Q

   model = ... # The pre-trained float model that needs to be quantified

   Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig) # EMA is a built-in QConfig for QAT
   
   for _ in range(...):
       train(model)

   Q.quantize(model)

#. :ref:`module-guide` ，并按照正常的浮点模型方式进行训练，得到预训练模型；
#. 使用 :func:`~.quantization.quantize_qat` 将 Float 模型转换为 QFloat 模型，
   这一步会基于量化配置 :class:`~.quantization.QConfig` 设置好 Observer 和 FakeQuantize 算子
   （在 MegEngine 中提供了常见的 QConfig :ref:`预设 <qconfig-list>`, 这里使用了 EMA 算法）；
#. 使用 QFloat 模型继续训练（微调），此时 Obersever 统计信息, FakeQuantize 进行伪量化；
#. 使用 :func:`~.quantization.quantize` 将 QFloat 模型转换为 Q 模型，这一步也叫 “真量化”（相较于伪量化）。
   此时网络无法再进行训练，网络中的算子都会转换为低比特计算方式，即可用于部署了。

.. mermaid::
   :align: center
   :caption: 此处为标准量化流程，实际使用时也可有灵活的变化

   flowchart LR
      PFM[Pre-trained Float Model] --> |quantize_qat| QFM[Pre-trained QFloat Model]
      QFM --> |train| FQFM[Fine-tuned QFloat Model]
      FQFM --> |quantize| QM[Q Model]

.. seealso::

   * 我们也可以使用 Calibration 后量化方案，需准备校准数据集（参考代码示范）；
   * MegEngine 的量化模型可被直接导出用于推理部署，参考 :ref:`dump` 。

   完整的 MegEngine 模型量化代码示范可在 :models:`official/quantization` 找到。

.. note::

   从宏观上看，量化是在 Model 级别之间的转换操作，但掰开细节，则都是对 Module 的处理。

   对应 Float, QFloat 和 Q Model, MegEngine 中的 Module 可被整理成以下三种：

   #. 进行正常浮点运算的默认 :class:`~.module.Module` （也即 Float Module ）
   #. 带有 Observer 和 FakeQuantize 算子的 :class:`.qat.QATModule`
   #. 无法训练、专门用于部署的 :class:`.quantized.QuantizedModule`

   对于其中比较常见的可以被量化的算子，分别有同名的实现如 ——

   * :class:`.module.Linear`, :class:`.module.qat.Linear` 和 :class:`.module.quantized.Linear`
   * :class:`.module.Conv2d`, :class:`.module.qat.Conv2d` 和 :class:`.module.quantized.Conv2d`

   对 Module 的处理用户无需感知，通过调用模型转换接口 :func:`~.quantization.quantize_qat` 和 :func:`~.quantization.quantize`,
   框架会完成相应算子的批量替换操作，感兴趣的用户可以阅读相应的源码逻辑，
   在 :ref:`module-convert` 小节中也会进行更具体的介绍。
  
.. _qconfig-guide:

量化配置 QConfig 说明
---------------------

:class:`~.quantization.QConfig` 包括 :class:`~.quantization.Observer` 和 :class:`~.quantization.FakeQuantize` 两部分，用户可 1.使用预设 2.自定义配置。

.. mermaid::
   :align: center

   flowchart LR
      FM[Float Model] --> QC{QConfig}
      QC -.- |Observer| QFM[QFloat Model]
      QC -.- |FakeQuantize| QFM[QFloat Model]

使用预设配置
~~~~~~~~~~~~

MegEngine 中提供了类似 ``ema_fakequant_qconfig`` 这样的预设，可用作 :func:`~.quantization.quantize_qat` 的 ``qconfig``:

>>> import megengine.quantization as Q
>>> Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

实际上它等同于使用以下 :class:`~.quantization.Qconfig` （以下即源码写法），以进行量化感知训练：

.. code-block::

    ema_fakequant_qconfig = QConfig(
        weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
        act_observer=partial(ExponentialMovingAverageObserver, dtype="qint8", narrow_range=False),
        weight_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=True),
        act_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=False),
    )

这里使用了两种 Observer 来统计信息，而 FakeQuantize 使用了默认的算子。

如果仅做后量化，或者说 Calibration, 由于无需进行 FakeQuantize, 故而其 ``fake_quant`` 属性为 None 即可：

.. code-block::

    calibration_qconfig = QConfig(
        weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
        act_observer=partial(HistogramObserver, dtype="qint8", narrow_range=False),
        weight_fake_quant=None,
        act_fake_quant=None,
    )

.. seealso::

   * 这里的 ``calibration_qconfig`` 也是可以直接使用的 Qconfig 预设配置；
   * 所有可用的 Qconfig 预设可以在 :ref:`量化 API 参考 <qconfig-list>` 中找到。

自定义 Observer 和 FakeQuantize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

除了使用预设配置，用户也可以根据需要灵活选择 Observer 和 FakeQuantize, 实现自己的 QConfig.

.. seealso::

   * Observer 举例：:class:`~.quantization.MinMaxObserver` / :class:`~.HistogramObserver` / :class:`~.ExponentialMovingAverageObserver` ...
   * FakeQuantize 举例：:class:`~.FakeQuantize` / :class:`~.TQT` / :class:`~.LSQ` ...
   * 所有可选的 Observer 和 FakeQuantize 已经列举在 :ref:`量化 API 参考 <qconfig-obsever>` 页面。

.. note::

   在实际使用过程中，可能需要在训练时让 Observer 统计并更新参数，但是在推理时则停止更新。
   Observer 和 FakeQuantize 自身都支持 :meth:`~.quantization.Observer.enable` 
   和 :meth:`~.quantization.Observer.disable` 方法，且 Observer 会在模型调用 :meth:`~.module.Module.train` 
   和 :meth:`~.module.Module.eval` 方法时自动分别调用对应的 ``Observer.enable/disable`` 方法。

   一般在进行数据校准时，会先执行 ``net.eval()`` 保证网络的参数不被更新，
   然后再调用 :func:`~.quantization.enable_observer` 函数来手动开启 Module 中 Observer 的统计修改功能
   （即先全局关闭，再开启特定的部分）：

   .. code-block:: python

      def calculate_scale(data, target):
          model.eval()  # all model observers are disabled now
          enable_observer(model)
          ...

   注意这些开关处理都是递归进行的。类似接口还有 :func:`~.quantization.disable_observer`, :func:`~.quantization.enable_fake_quant`,
   :func:`~.quantization.disable_fake_quant` 等，可在 :ref:`quantize-operation` 中找到。

.. _module-convert:

模型转换模块与相关基类
----------------------

QConfig 提供了一系列如何对模型做量化的接口，而要使用这些接口，
需要网络的 Module 能够在 forward 时给权重、激活值加上 Observer 和进行 FakeQuantize.
转换模块的作用就是将模型中的普通 :class:`~.module.Module` 替换为支持这一系列操作的 :class:`~.module.qat.QATModule` ，
并能支持进一步替换成无法训练、专用于部署的 :class:`~.module.quantized.QuantizedModule` .

这三种 Module 与 Model 对应，通过转换接口可以依次替换为不同实现的同名 Module.

.. mermaid::
   :align: center
   :caption: 以 Conv2d 为例，从 Moudle 到 QATModule 再到 QuantizedModule.

   flowchart LR
       M[module.Conv2d] -- quantize_qat --> QATM[module.qat.Conv2d] -- quantize --> QM[module.quantized.Conv2d]

同时考虑到量化与推理优化时常用的算子融合（Fuse）技术高度关联，MegEngine 中提供了一系列预先融合好的 Module，
比如 :class:`~.module.ConvRelu2d` 、 :class:`~.module.ConvBn2d` 和 :class:`~.module.ConvBnRelu2d` 等。
显式地使用融合算子可以保证过程更加可控，其对应的 QuantizedModule 版本都会直接调用底层实现好的融合算子；
否则框架需要自己根据网络结构进行自动匹配和融合优化。
这样实现的缺点在于用户在使用时需要修改原先的网络结构，使用融合好的 Module 搭建网络。
而好处则是用户能更直接地控制网络如何转换，比如同时存在需要融合和不需要融合的 Conv 算子，
相比提供一个冗长的白名单，我们更倾向于在网络结构中显式地控制；而一些默认会进行转换的算子，
也可以通过 :meth:`~.module.Module.disable_quantize` 方法来控制其不进行转换（下面有举例）。

除此之外还提供专用于量化的 :class:`~.module.QuantStub` 、 :class:`~.module.DequantStub` 等辅助模块。

转换的原理很简单，就是将父 Module 中可被量化（Quantable）的子 Module 替换为对应的新 Module. 
但是有一些 Quantable Module 还包含 Quantable 子 Module，比如 ConvBn 就包含一个 Conv2d 和一个 BatchNorm2d，
转换过程并不会对这些子 Module 进一步转换，原因是父 Module 被替换之后，
其 forward 计算过程已经完全不同了，不会再依赖于这些子 Module.

.. note::

    如果需要使一部分 Module 及其子 Module 保留 Float 状态，不进行转换，
    可以使用 :meth:`~.module.Module.disable_quantize` 来处理。
    比如当你发现对 fc 层进行量化后，模型会掉点，则可以关闭该层的量化处理：

    >>> model.fc.disable_quantize()

    该接口也可以被当作装饰器进行使用，方便对多个 Module 进行处理。

..  warning::

    如果网络结构中涉及一些二元及以上的 ElementWise 操作符，比如加法乘法等，
    由于多个输入各自的 scale 并不一致，必须使用量化专用的算子，并指定好输出的 scale. 
    实际使用中只需要把这些操作替换为 :class:`~.module.Elemwise` 即可，
    比如 ``self.add_relu = Elemwise("FUSE_ADD_RELU")``

    目前支持的量化 Elemwise 算子可在 :src:`dnn/scripts/opr_param_defs.py` 中找到：

    .. code-block:: python

       pdef('ElemwiseMultiType').add_enum(
           'Mode',
           # ...
           Doc('QFUSE_ADD_RELU = 7', 'Fused elemwise add two quantized int8 followed'
               ' by ReLU and typecvt to specified dtype'),
           # ...
       )

    注意：在量化模型过程中，使用 Elemwise 算子不用加上前置 Q.

    另外由于转换过程修改了原网络结构，模型保存与加载无法直接适用于转换后的网络，
    读取新网络保存的参数时，需要先调用转换接口得到转换后的网络，
    才能用 :meth:`~.module.Module.load_state_dict` 将参数进行加载。

ResNet 实例讲解
---------------

下面我们以 ResNet18 为例来讲解量化的完整流程。主要分为以下几步：

#. 修改网络结构，使用已经融合好的 ConvBn2d、ConvBnRelu2d、ElementWise 代替原先的 Module.
   在正常模式下预训练模型，并在每轮迭代保存网络检查点；
#. 调用 :func:`~.quantization.quantize_qat` 转换模型，并进行量化感知训练微调（或校准，取决于 QConfig）；
#. 调用 :func:`~.quantization.quantize` 转换为量化模型，导出模型用于后续模型部署。

.. seealso::

   这里对代码进行了简化，完整的 MegEngine 官方量化示例代码见： :models:`official/quantization`

训练 Float 模型
~~~~~~~~~~~~~~~

我们修改了模型结构中的一些子 Module, 将原先单独的 ``Conv``, ``BN``, ``ReLU`` 替换为融合后的可被量化的 Module.

* 修改前的模型结构： :models:`official/vision/classification/resnet/model.py`
* 修改后的模型结构： :models:`official/quantization/models/resnet.py`

以 ``BasicBlock`` 模块的修改前后作为例子对比：

.. code-block:: python

   class BasicBlock(M.Module):
         def __init__(self, in_channels, channels):
            super().__init__()
            self.conv1 = M.Conv2d(in_channels, channels, 3, 1, padding=dilation, bias=False)
            self.bn1 = M.BatchNorm2d
            self.conv2 = M.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
            self.bn2 = M.BatchNorm2d
            self.downsample = (
               M.Identity()
               if in_channels == channels and stride == 1
               else M.Sequential(
               M.Conv2d(in_channels, channels, 1, stride, bias=False)
               M.BatchNorm2d
            )

         def forward(self, x):
            identity = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            identity = self.downsample(identity)
            x = F.relu(x + identity)
            return x

.. code-block:: python
   :emphasize-lines: 4,5,9,11

   class BasicBlock(M.Module):
         def __init__(self, in_channels, channels):
            super().__init__()
            self.conv_bn_relu1 = M.ConvBnRelu2d(in_channels, channels, 3, 1, padding=dilation, bias=False)
            self.conv_bn2 = M.ConvBn2d(channels, channels, 3, 1, padding=1, bias=False)
            self.downsample = (
               M.Identity()
               if in_channels == channels and stride == 1
               else M.ConvBn2d(in_channels, channels, 1, 1, bias=False)
            )
            self.add_relu = M.Elemwise("FUSE_ADD_RELU")

         def forward(self, x):
            identity = x
            x = self.conv_bn_relu1(x)
            x = self.conv_bn2(x)
            identity = self.downsample(identity)
            x = self.add_relu(x, identity)
            return x

然后对该模型进行若干轮迭代训练，并保存检查点，这里省略细节：

.. code-block:: python

    for step in range(0, total_steps):
        # Linear learning rate decay
        epoch = step // steps_per_epoch
        learning_rate = adjust_learning_rate(step, epoch)

        image, label = next(train_queue)
        image = tensor(image.astype("float32"))
        label = tensor(label.astype("int32"))

        n = image.shape[0]

        loss, acc1, acc5 = train_func(image, label, net, gm)  # traced
        optimizer.step().clear_grad()

        # Save checkpoints

.. seealso::

   * Train - :models:`official/quantization/train.py`

转换成 QFloat 模型
~~~~~~~~~~~~~~~~~~

调用 :func:`~.quantization.quantize_qat` 来将网络转换为 QFloat 模型:

.. code-block:: python

   from megengine.quantization import ema_fakequant_qconfig, quantize_qat

   model = ResNet18()

   # QAT
   quantize_qat(model, ema_fakequant_qconfig)

   # Or Calibration:
   # quantize_qat(model, calibration_qconfig)

读取预训练 Float 模型保存的检查点，继续使用上面相同的代码进行微调 / 校准。

.. code-block:: python

   if args.checkpoint:
       logger.info("Load pretrained weights from %s", args.checkpoint)
       ckpt = mge.load(args.checkpoint)
       ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
       model.load_state_dict(ckpt, strict=False)

   # Fine-tune / Calibrate with new traced train_func
   # Save checkpoints

最后也需要保存此时 QFloat 模型的检查点，以便在测试和推理进行 QFloat 模型的加载和转换。

.. warning::

   * 需要将原始 Float 模型转换为 QFloat 模型之后再加载检查点；
   * 如果这两次训练全在同一个脚本中执行，那么训练的 traced 函数需要用不一样的，
     因为此时模型的参数变化了，需要重新进行编译。

.. seealso::

   * Finetune - :models:`official/quantization/finetune.py`
   * Calibration - :models:`official/quantization/calibration.py`

转换成 Q 模型
~~~~~~~~~~~~~

将 QFloat 模型转换为 Q 模型并导出，共包括以下几步：

.. code-block:: python
   :emphasize-lines: 10

   from megengine.quantization import quantize

   @jit.trace(capture_as_const=True)
   def infer_func(processed_img):
       model.eval()
       logits = model(processed_img)
       probs = F.softmax(logits)
       return probs

   quantize(model)

   processed_img = transform.apply(image)[np.newaxis, :]
   processed_img = processed_img.astype("int8")
   probs = infer_func(processed_img)

   infer_func.dump(output_file, arg_names=["data"])

#. 定义 trace 函数，打开 ``capture_as_const`` 以进行模型导出；
#. 调用 :func:`~.quantization.quantize` 将 QAT 模型转换为 Quantized 模型；
#. 准备数据并执行一次推理，调用 :meth:`~.trace.dump` 将模型导出。

至此便得到了一个可用于部署的量化模型。

.. seealso::

   * Inference and dump - :models:`official/quantization/inference.py`

