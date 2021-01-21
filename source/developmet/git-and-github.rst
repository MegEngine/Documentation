.. _git-and-github:

==================
Git 与 GitHub 基础
==================

本页面简要描述了适用于 MegEngine_ 的 Git_ 和 GitHub_ 工作流程。

.. _MegEngine: https://github.com/MegEngine/MegEngine
.. _git: https://git-scm.com/
.. _github: https://github.com/

安装 Git
--------

Git_ 是一个分布式版本控制系统，想要进一步了解，
请参考 `Git 是什么 <https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-Git-%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F>`_ 。

在你开始使用 Git 前，需要将它安装在你的计算机上。即便已经安装，最好将它升级到最新的版本。

以 Ubuntu（基于 Debian 的 Linux 发行版）为例，你可以使用 ``apt`` 来安装 Git:

.. code-block:: shell

   sudo apt install git-all

如果你使用的是其它操作系统，
请参考 `安装 Git <https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-%E5%AE%89%E8%A3%85-Git>`_ 。

安装完成后，你可以通过 `git version` 命令来查看已安装的 Git 版本。

将 MegEngine 代码拷贝到本地
---------------------------

请在你的工作区目录使用以下命令：

.. code-block:: shell

   git clone git@github.com:MegEngine/MegEngine.git

命令执行完成后，你将在本地得到一个新的 MegEngine 代码树的拷贝。

有的时候你会希望从远端主分支拉取最新的代码，请在你的工作区目录使用以下命令：

.. code-block:: shell

   cd MegEngine
   git fetch
   git merge --ff-only
