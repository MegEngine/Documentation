.. py:module:: megengine.utils
.. currentmodule:: megengine.utils

==========
Utils 模块
==========

.. py:module:: megengine.utils.comp_graph_tools
.. currentmodule:: megengine.utils.comp_graph_tools

Computing Graph Tools
---------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   get_dep_vars
   get_owner_opr_inputs
   get_owner_opr_type
   get_opr_type
   graph_traversal
   get_oprs_seq
   replace_vars
   replace_oprs
   set_priority_to_id
   GraphInference

.. py:module:: megengine.utils.hook
.. currentmodule:: megengine.utils.hook

Hook
----
.. autosummary::
   :toctree: api
   :nosignatures:

   HookHandler

.. py:module:: megengine.utils.network
.. currentmodule:: megengine.utils.network

Network 
-------
.. autosummary::
   :toctree: api
   :nosignatures:

   Network

.. rubric:: 方法
.. autosummary::
   :toctree: api
   :nosignatures:

   Network.load
   Network.dump
   Network.make_const
   Network.make_input_node
   Network.add_output
   Network.remove_output
   Network.add_dep_oprs
   Network.modify_opr_names
   Network.reset_batch_size
   Network.replace_vars
   Network.replace_oprs
   Network.get_opr_by_type
   Network.get_opr_by_name
   Network.get_var_by_name
   Network.get_var_receive_oprs
   Network.get_dep_oprs

.. rubric:: 属性
.. autosummary::
   :toctree: api
   :nosignatures:
 
   Network.opr_filter
   Network.var_filter
   Network.params_filter
   Network.data_providers_filter
   Network.dest_vars
   Network.all_oprs
   Network.all_vars
   Network.all_oprs_dict
   Network.all_vars_dict
   
.. rubric:: Convert 
.. autosummary::
   :toctree: api
   :nosignatures:

   as_varnode
   as_oprnode

NodeFilter
~~~~~~~~~~

.. autosummary::
   :toctree: api
   :nosignatures:
 
   NodeFilter
   NodeFilter.type
   NodeFilter.check_type
   NodeFilter.not_type
   NodeFilter.param_provider
   NodeFilter.data_provider
   NodeFilter.name
   NodeFilter.has_input
   NodeFilter.as_list
   NodeFilter.as_unique
   NodeFilter.as_dict
   NodeFilter.as_count

Others
~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   NodeFilterType
   NodeFilterNotType
   NodeFilterCheckType
   NodeFilterHasInput
   NodeFilterName

.. py:module:: megengine.utils.network_node
.. currentmodule:: megengine.utils.network_node

Network Node
------------
.. autosummary::
   :toctree: api
   :nosignatures:

   NetworkNode
   VarNode
   OpNode

更多 Network Node 请查看源码。

.. py:module:: megengine.utils.module_stats
.. currentmodule:: megengine.utils.module_stats

Module Stats
------------
.. autosummary::
   :toctree: api
   :nosignatures:

   module_stats


.. py:module:: megengine.utils.profiler
.. currentmodule:: megengine.utils.profiler

Profiler
--------
.. autosummary::
   :toctree: api
   :nosignatures:

   Profiler
   profile

默认常量
~~~~~~~~
.. autoattribute:: Profiler.CHROME_TIMELINE
.. autoattribute:: Profiler.COMMAND
.. autoattribute:: Profiler.OPERATOR
.. autoattribute:: Profiler.TENSOR_LIFETIME
.. autoattribute:: Profiler.TENSOR_PROP
.. autoattribute:: Profiler.SYNC
.. autoattribute:: Profiler.SCOPE
.. autoattribute:: Profiler.ALL

