.. _comparison-batch-norm:

===================
BatchNorm 差异对比
===================

.. panels::

  torch.nn.BatchNorm2d
  ^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.BatchNorm2d(
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
     )

  更多请查看 :py:class:`torch.nn.BatchNorm2d`.

  ---

  megengine.module.BatchNorm2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.BatchNorm2d(
         num_features,
         eps=1e-05,
         momentum=0.9,
         affine=True,
         track_running_stats=True, 
         freeze=False,
         ** kwargs
     )

  更多请查看 :py:class:`megengine.module.BatchNorm2d`.

功能差异
--------

momentum 差异
~~~~~~~~~~~~~

.. warning:: 
   
   MegEngine 的 ``momentum`` 参数默认值为 0.9, 而 PyTorch 的默认值为 0.1.
   在实际计算时效果一致，这表明该参数在 MegEngine 中的含义与 PyTorch 中的含义不同。

.. admonition:: running_mean 和 running_var 的计算方式

   MegEngine 中 ``running_mean`` 和 ``running_var`` 的更新公式如下：

   .. math::

      \begin{aligned}

      \textrm{running_mean} = &\textrm{momentum} \times \textrm{running_mean} \\
                              &+ (1 - \textrm{momentum}) \times \textrm{batch_mean}

      \end{aligned}

   ``running_var`` 的更新过程与上同理，MegEngine 的 ``momentum`` 的含义更符合其惯性的实际语义，
   而 PyTorch 的 ``momentum`` 参数是指数加权平均的衰减率，即 ``1 - momentum`` （此处指 MegEngine 中的 ``momentum`` ）。

冻结参数
~~~~~~~~

:py:class:`megengine.module.BatchNorm2d` 支持 ``freeze`` 参数，用于冻结 BN 层的参数。

在 Pytorch 中可能需要使用类似的方法进行冻结：

.. code-block:: python

   for child in model.children():
       if isinstance (child, torch.nn.BatchNorm2d):
           for param in child.parameters():
               param.requires_grad = False

而在 MegEngine 中，只需要将 ``freeze`` 参数设置为 ``True`` 即可。
