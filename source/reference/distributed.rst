.. py:module:: megengine.distributed
.. currentmodule:: megengine.distributed

=====================
megengine.distributed
=====================

分组（Group）
-------------
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Group
   init_process_group
   new_group
   is_distributed
   get_backend
   get_client
   get_mm_server_addr
   get_py_server_addr
   get_world_size
   group_barrier

运行器（Launcher）
------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   launcher

客户端与服务端(C/S)
-------------------
.. autosummary::
   :toctree: api
   :nosignatures:
  
   Client
   Server

辅助功能（Helper）
------------------
.. autosummary::
   :toctree: api
   :nosignatures:
  
   bcast_list_
   synchronized
   make_allreduce_cb
   helper.AllreduceCallback
