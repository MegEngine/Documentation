.. py:module:: megengine.data
.. currentmodule:: megengine.data

============
数据（Data）
============

.. py:module:: megengine.data.dataset
.. currentmodule:: megengine.data.dataset

自定义数据集（Dataset）
-----------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   Dataset
   ArrayDataset
   StreamDataset

计算机视觉经典数据集
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   ImageFolder 
   MNIST
   CIFAR10
   CIFAR100
   PascalVOC
   COCO
   Cityscapes
   Objects365
   ImageNet

.. py:module:: megengine.data.sampler
.. currentmodule:: megengine.data

采样器（Sampler）
-----------------
.. autosummary::
   :toctree: api
   :nosignatures:

   Sampler
   MapSampler
   StreamSampler
   SequentialSampler
   RandomSampler
   ReplacementSampler
   Infinite

.. py:module:: megengine.data.transform
.. currentmodule:: megengine.data.transform

常见变换（Transform）
---------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   VisionTransform
   ToMode
   Compose
   TorchTransformCompose
   Pad
   Resize
   ShortestEdgeResize
   RandomResize
   RandomCrop
   RandomResizedCrop
   CenterCrop
   RandomHorizontalFlip
   RandomVerticalFlip
   Normalize
   GaussianNoise
   BrightnessTransform
   SaturationTransform
   ContrastTransform
   HueTransform
   ColorJitter
   Lighting

.. py:module:: megengine.data.collator
.. currentmodule:: megengine.data

数据合并器（Collator）
----------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   Collator

.. py:module:: megengine.data.dataloader
.. currentmodule:: megengine.data

数据加载器（DataLoader）
------------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   DataLoader


