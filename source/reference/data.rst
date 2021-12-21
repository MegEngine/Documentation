.. py:module:: megengine.data
.. currentmodule:: megengine.data

==============
megengine.data
==============

>>> import megengine.data as data
>>> import megengine.data.transform as T

DataLoader
----------
.. autosummary::
   :toctree: api
   :nosignatures:

   DataLoader

.. py:module:: megengine.data.dataset
.. currentmodule:: megengine.data.dataset

Dataset
-------
.. autosummary::
   :toctree: api
   :nosignatures:

   Dataset
   ArrayDataset
   StreamDataset

Vision Dataset
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   VisionDataset
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

Sampler
-------
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

Transform
---------
.. autosummary::
   :toctree: api
   :nosignatures:

   Transform
   PseudoTransform

Vision Transform
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   VisionTransform
   Compose
   TorchTransformCompose
   ToMode
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

Collator
--------
.. autosummary::
   :toctree: api
   :nosignatures:   

   Collator