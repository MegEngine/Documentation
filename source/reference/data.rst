.. py:module:: megengine.data
.. currentmodule:: megengine.data

==============
megengine.data
==============
DataLoader
----------
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   DataLoader

.. py:module:: megengine.data.dataset
.. currentmodule:: megengine.data.dataset

Dataset
-------
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Dataset
   ArrayDataset
   StreamDataset

Vision Dataset
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

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
   :template: autosummary/api-class.rst

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
   :template: autosummary/api-class.rst

   Transform
   PseudoTransform

Vision Transform
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

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
   :template: autosummary/api-class.rst

   Collator

