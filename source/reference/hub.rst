.. py:module:: megengine.hub
.. currentmodule:: megengine.hub

=============
megengine.hub
=============

查询/使用预训练模型
-------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   list
   load
   help
   load_serialized_obj_from_url
   pretrained
   import_module

.. py:module:: megengine.hub.fetcher
.. currentmodule:: megengine.hub.fetcher

Fetcher
--------

.. autosummary::
   :toctree: api
   :nosignatures:

   synchronized
   GitSSHFetcher
   GitHTTPSFetcher

.. py:module:: megengine.hub.tools
.. currentmodule:: megengine.hub.tools

Tools
--------

.. autosummary::
   :toctree: api
   :nosignatures:

   load_module
   check_module_exists
   cd
   
.. py:module:: megengine.hub.exceptions
.. currentmodule:: megengine.hub.exceptions

异常处理
--------

.. autosummary::
   :toctree: api
   :nosignatures:

   FetcherError
   GitCheckoutError
   GitPullError
   InvalidGitHost
   InvalidProtocol
   InvalidRepo

