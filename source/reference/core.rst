==============
megengine.core
==============
.. py:module:: megengine.core
.. currentmodule:: megengine.core

.. code-block:: python

   """ 
   Users should never --
       import megengine.core
   But MegEngine developers should know what is inside. 
   """

.. warning::

   我们不承诺 core 模块中 API 的兼容性和稳定性。

tesnor
~~~~~~
.. currentmodule:: megengine.core.tensor

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   core.OpBase
   core.TensorBase
   core.TensorWrapperBase
   array_method.ArrayMethodMixin
   dtype.QuantDtypeMeta
   dtype.get_dtype_bit
   dtype.is_lowbit
   dtype.is_bfloat16
   dtype.create_quantized_dtype
   dtype.quint8
   dtype.qint8
   dtype.qint32
   dtype.quint4
   dtype.qint4
   dtype.convert_to_quint8
   dtype.convert_to_qint8
   dtype.convert_from_qint8
   dtype.convert_to_qint32
   dtype.convert_from_qint32
   dtype.convert_to_quint4
   dtype.convert_from_quint4
   dtype.convert_to_qint4
   dtype.convert_from_qint4
   dtype.convert_from_quint8
   indexing.remove_ellipsis
   indexing.check_bool_index
   indexing.unpack_getitem
   indexing.try_condtake
   indexing.getitem
   indexing.setitem
   megbrain_graph.set_priority_to_id
   megbrain_graph.Graph
   megbrain_graph.VarNode
   megbrain_graph.OpNode
   megbrain_graph.optimize_for_inference
   megbrain_graph.modify_opr_algo_strategy_inplace
   megbrain_graph.dump_graph
   megbrain_graph.load_graph
   megbrain_graph.apply_normal_varnode
   megbrain_graph.apply_backward_varnode
   megbrain_graph.input_callback
   megbrain_graph.InputNode
   megbrain_graph.output_callback
   megbrain_graph.OutputNode
   megbrain_graph.ValueOutputNode
   megbrain_graph.TensorAttr
   megbrain_graph.AttrOutputNode
   megbrain_graph.VirtualDepNode

ops
~~~
.. currentmodule:: megengine.core.ops

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   special.Const

autodiff
~~~~~~~~
.. currentmodule:: megengine.core.autodiff

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   grad.get_grad_managers
   grad.GradKey
   grad.Grad
   grad.Function

