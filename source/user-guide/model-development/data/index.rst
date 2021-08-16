.. _data-guide:
.. currentmodule:: megengine

===========================
使用 Data 构建输入 Pipeline 
===========================
MegEngine 中的 :py:mod:`data` 子包提供了用于处理数据（数据集）的原语，
其中 :py:class:`megengine.data.DataLoader` 被用于加载批数据，本质上将用于生成一个可迭代的对象，
负责从 :py:class:`~.Dataset` 描述的数据集中随机返回批量大小 ``batch_size`` 的数据。
简而言之， ``Dataset`` 告诉 ``DataLoader`` 如何将单个训练或测试数据加载到内存中，
而 ``DataLoader`` 负责按照给定的配置获取分批的数据，方便进行后续训练和测试。

上面的介绍中隐藏了一些细节，如果你想要更加正确 & 高效地构建输入 Pipeline, 建议阅读完这一章节的内容。

.. seealso::

   该部分功能的主体设计与 PyTorch 提供的
   `torch.utils.data <https://pytorch.org/docs/stable/data.html>`_ 类似。

.. _dataloader-guide:

使用 DataLoader 加载数据
------------------------

:py:class:`~.DataLoader` 类的签名如下：

.. class:: DataLoader(dataset, sampler=None, transform=None, collator=None, 
                      num_workers=0, timeout=0, timeout_event=raise_timeout_error, divide=False)
   :noindex:

以下选项可作为 :py:class:`~.DataLoader` 类的构造函数参数进行灵活配置：

.. toctree::
   :maxdepth: 1
   
   dataset
   sampler
   transform
   collator

.. note::

   以模型训练为例，在 MegEngine 中输入数据的 Pipeline 为：

   #. 创建一个 Dataset 对象；
   #. 按需创建 Sampler, Transform 和 Collator 对象；
   #. 创建一个 DataLoader 对象；
   #. 迭代这个 DataLoader 对象，将数据分批加载到模型中进行训练；
   
   需要注意以下几点：

   * 如果不自定义配置，用户应当清楚在使用默认参数的情况下 DataLoader 的处理逻辑；
   * 同理，模型的验证和测试也可以使用各自的 DataLoader 完成数据部分的加载；

.. _load-image-data-example:

举例：加载图像分类数据
----------------------

下面我们以加载图像分类数据的基本流程作为简单举例 ——

#. 假设图像数据按照一定的规则放置于同一目录下（通常数据集主页会对目录组织和文件命名规则进行介绍）。
   要创建对应的数据加载器，首先需要一个继承自 :py:class:`~.Dataset` 的类。
   虽然对于 NumPy ndarray 数据，MegEngine 中提供了 :py:class:`~.ArrayDataset` 实现。
   但更标准的做法应当是创建一个自定义的数据集：

   .. code-block:: python

      import cv2
      import numpy as np
      import megengine
      from megengine.data.dataset import Dataset

      class CustomImageDataset(Dataset):
          def __init__(self, image_folder):
              # get all mapping indice
              self.image_folder = image_folder
              self.image_list = os.listdir(image_folder)

          # get the sample
          def __getitem__(self, idx):
              # get the index
              image_file = self.image_list[idx]

              # get the data
              # in this case we load image data and convert to ndarray
              image = cv2.imread(self.image_folder + image_file, cv2.IMREAD_COLOR)
              image = np.array(image)

              # get the label
              # in this case the label was noted in the name of the image file
              # ie: 1_image_28457.png where 1 is the label 
              # and the number at the end is just the id or something
              target = int(image_file.split("_")[0])

              return image, target

          def __len__(self):
              return len(self.images)

   要获取示例图像，可以创建一个数据集对象，并将示例索引传递给 ``__getitem__`` 方法，
   然后将返回图像数组和对应的标签，例如：

   .. code-block:: python

      dataset = CustomImageDataset("/path/to/image/folder")
      data, sample = dataset.__getitem__(0) # dataset[0]

#. 现在我们已经预先创建了能够返回一个样本及其标签的类 ``CustomImageDataset``, 
   但仅依赖 ``Dataset`` 本身还无法实现自动分批、乱序、并行等功能；
   我们必须接着创建 ``DataLoader``, 它通过其它的参数配置项围绕这个类“包装”，
   可以按照我们的要求从数据集类中返回整批样本。

   .. code-block:: python

      from megengine.data.transform import ToMode
      from megengine.data import DataLoader, RandomSampler

      dataset = YourImageDataset("/path/to/image/folder")

      # you can implement the function to randomly split your dataset
      train_set, val_set, test_set = random_split(dataset)

      # B is your batch-size, ie. 128
      train_dataloader = DataLoader(train_set,
            sampler=RandomSampler(train_set, batch_size=B),
            transform=ToMode('CHW'),
      )

   注意到在上面的代码中，我们还用到了 ``Sampler`` 来决定数据加载（抽样）顺序，
   用到了 ``Transform`` 来对加载后的数据进行一些变换处理，这还不是全部可配置项，
   在后续小节我们会进行更加详细的介绍。

#. 现在我们已经创建了数据加载器并准备好训练！例如像这样：

   .. code-block:: python

      for epoch in range(epochs):

          for images, targets in train_dataloder:
              # now 'images' is a batch containing B samples
              # and 'targets' is a batch containing B targets 
              # (of the images in 'images' with the same index

              # remember to convert data to tensor
              images = megengine.Tensor(images)
              targets = megengine.Tensor(targets)

              # train function
              # ...

#. 成功地获取到批数据后，关于模型如何训练和测试的后续流程就不在这里介绍了。

.. seealso::

   * 在 MegEngine 新手入门板块中提供了完整的基于 MNIST 和 CIFAR10 数据集的模型训练与测试教程；
   * 在 MegEngine 官方模型库 `Models <https://github.com/MegEngine/Models>`_ 中可以找到更多参考代码。
