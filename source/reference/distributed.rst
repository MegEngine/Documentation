.. py:module:: megengine.distributed
.. currentmodule:: megengine.distributed

=====================
megengine.distributed
=====================

>>> import megengine.distributed as dist

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/mprop.rst

   backend

分组（Group）
-------------
.. autosummary::
   :toctree: api
   :nosignatures:

   Server
   Group
   init_process_group
   new_group
   group_barrier
   override_backend
   is_distributed
   get_backend
   get_client
   get_mm_server_addr
   get_py_server_addr
   get_rank
   get_world_size


运行器（Launcher）
------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   launcher

辅助功能（Helper）
------------------
.. autosummary::
   :toctree: api
   :nosignatures:
  
   bcast_list_
   synchronized
   make_allreduce_cb
   helper.AllreduceCallback
   helper.param_pack_split
   helper.param_pack_concat
   helper.pack_allreduce_split
