.. _comparison-pixel-shuffle:

=============================
PixelShuffle 差异对比
=============================

.. panels::

  torch.nn.PixelShuffle
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.PixelShuffle(
        upscale_factor
     )

  更多请查看 :py:class:`torch.nn.PixelShuffle`.

  ---

  megengine.module.PixelShuffle
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.PixelShuffle(
       upscale_factor
     )

  更多请查看 :py:class:`megengine.module.PixelShuffle`.




参数无差异，功能、用法相同。


.. code-block::: python

    import torch    

    pixel_shuffle = nn.PixelShuffle(3)
    input = torch.randn(1, 9, 4, 4)
    output = pixel_shuffle(input)

.. code-block::: python

    import megengine

    pixel_shuffle = megengine.module.PixelShuffle(3)
    input = megengine.random.normal(size=(1,9,4,4))
    output = pixel_shuffle(input)
