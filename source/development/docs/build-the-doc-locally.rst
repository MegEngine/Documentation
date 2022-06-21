.. _how-to-build-the-doc-locally:

如何在本地构建与预览文档
========================

.. warning::

   文档构建有 FULL 和 MINI 两种不同的模式，可以通过配置环境变量 ``MGE_DOC_MODE`` 来决定具体的行为，
   该环境变量提供以下三种选项：

   ``AUTO`` （默认）
     自动探测 MegEngine 包是否可用，如可用则进入 FULL 模式，否则进入 MINI 模式；

   ``MINI``
     构建除 MegEngine API Reference 外的文档，**不依赖于 MegEngine 源代码** 。

   ``FULL``
     构建全部文档，包括 MegEngine API Reference, 需要设置好 MegEngine 路径。

.. note::

   * 下面以 Ubuntu 18.04 + Python 3.8 环境为例，向你展示从无到有构建 MegEngine 文档的过程；
   * 具有 ``sudo`` 特权的用户，可以使用 :docs:`scripts/bootstrap.sh` 脚本自动完成环境准备。

克隆文档源码到本地
------------------

将存储库克隆到本地（默认为 ``main`` 分支），确保目录下有 ``Makefile`` 文件：

.. code-block:: shell

   git clone https://github.com/MegEngine/Documentation
   cd Documentation

对于一些旧版本的 Git, 还需要用下面的命令从 GitHub LFS 服务器拉取对应文件：

.. code-block:: shell

   git lfs install
   git lfs pull

.. note::

   为确保正常克隆，需要在环境中安装好 LFS_ (Large File Storage) 插件。

.. _LFS: https://git-lfs.github.com/

初始化第三方依赖
----------------

.. code-block:: shell

  git submodule update --init --progress --depth=1 --recursive

这一步将会拉取文档所依赖的第三方子模块，比如文档所用的主题（在后面的步骤中会进行安装）。

.. _megengine-path:

指定 MegEngine 路径（可选）
---------------------------

在构建文档时，会尝试执行 ``import megengine`` 和 ``import megenginelite`` 
来判断使用 ``FULL`` 还是 ``MINI`` 模式。在检测到本地环境没有 MegEngine 时，
将采用 ``MINI`` 模式，该模式将跳过所有 API 文档页面的生成，
同时也会忽略掉一些警告信息的检查，因此它仅适用于修改幅度极其小的情况（如单页面，不存在交叉引用等）。
在绝大部分情况下，推荐使用 ``FULL`` 模式，以便进行完整的文档内容生成和语法、样式检查。

根据不同的需求，有两种方式将用于构建文档的 MegEngine 导入当前 Python 环境（任选其一即可）：

* 如果你是框架用户，不需要改动 MegEngine 源码，只想在本地完整地构建和预览所有文档的内容，
  或进行简单的增删查改，建议安装最新发布的 MegEngine 稳定版 Wheel 包。
  可以直接使用对应的 ``pip intall`` 命令将已经打包好的 MegEngine 安装到当前的 Python 环境中。
  ➡️  :ref:`了解如何进行使用 pip 安装 <install>` 。
* 如果你是研发人员，需要在指定的 MegEngine 分支源代码上生成对应文档，则需要克隆对应分支进行编译构建。
  通过 ``export PYTHONPATH/LITE_LIB_PATH`` 的形式来临时指定特定源代码路径，例如：

  .. code-block:: shell

     export PYTHONPATH=/data/MegEngine/imperative/python:$PYTHONPATH
     export PYTHONPATH=/data/MegEngine/lite/pylite$:$PYTHONPATH
     export LITE_LIB_PATH=/data/MegEngine/build/lite/liblite_shared.so  

  **注意一定要使用版本一致的 MegEngine 与 Lite** , 否则可能导致产生符号冲突。
  这种方式适合开发者需要同时对源码和文档进行维护的情况。➡️  :ref:`了解如何进行从源码构建 <install>` 。

安装 Sphinx 与 Pydata 主题
--------------------------

MegEngine 文档使用 Sphinx_ 进行整个网站的构建，请运行下面的指令，安装 Sphinx 和相关依赖：

.. _Sphinx: https://www.sphinx-doc.org

.. code-block:: shell

   python3 -m pip install -r requirements.txt

.. warning::

   MegEgnine 文档使用了 Fork 后修改过的
   `pydata-sphinx-theme <https://github.com/MegEngine/pydata-sphinx-theme/tree/dev>`_ 主题，
   如果你的本地环境已经存在该主题，可能需要提前删除该主题或使用额外的 Python 虚拟环境。

安装相关软件包
--------------

Pandoc 转换工具
~~~~~~~~~~~~~~~

nbsphinx_ 是 Sphinx 的一个插件，可以帮助我们对 ``.ipynb`` 格式的 Jupyter Notebook_ 文件进行解析。

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _Notebook: https://jupyter.org/

我们在安装依赖环境时已经安装好了 nbsphinx, 但还需要通过依赖项目 Pandoc_ 来支持转换 Markdown 格式。

.. _Pandoc: https://pandoc.org/

如果你使用的是是 Ubuntu（Debian）操作系统，可以直接使用 ``apt`` 命令进行安装 Pandoc:

.. code-block:: shell

   sudo apt install -y pandoc

如果你使用的是其它操作系统，想要安装 Pandoc，请参考 Pandoc 官方的 `Installing <https://pandoc.org/installing.html>`_ 页面。

Graphviz 绘图工具
~~~~~~~~~~~~~~~~~

Graphviz_ 是非常流行的图形可视化软件，在 MegEngine 文档中经常会用它制作一些可视化图片。

如果你使用的是 Ubuntu（Debian）操作系统，可以直接使用 ``apt`` 命令进行安装 Graphviz:

.. code-block:: shell

   sudo apt install -y graphviz

如果你使用的是其它操作系统，想要安装 Graphviz，请参考 Graphviz 官方的 `Download <https://graphviz.org/download/>`_ 页面。

.. _Graphviz: https://graphviz.org/

使用 Sphinx 进行文档构建
------------------------

#. 运行 ``make help`` 指令，可看到相应的帮助和参数信息；
#. 在文档目录下使用 ``make html`` 指令，会在 ``build`` 目录下生成 HTML 文件夹。
#. 文档生成成功后，打开 ``build/html/index.html`` 文件便可访问主页。

.. note::

   Sphinx 默认支持增量构建，当你再次执行 ``make html`` 时将仅对变化的文件进行更新；

.. warning::

   Sphinx 不会检测增量模式下非文档文件的更改，例如主题文件、静态文件和与 autodoc 一起使用的源代码；
   如果发现一些页面的元素仍被缓存而没有被更新，请尝试通过传入 ``-a`` 参数禁用增量模式（但构建速度会相应地变慢），
   或者通过 ``make clean`` 指令清除掉已经构建出的内容。

自动构建和实时预览页面
----------------------

你也可以使用 ``make livehtml`` 指令，在监测到文件变化时自动重新构建，而且可以通过浏览器进行实时的预览。
其中 ``HOST`` 参数默认为 ``127.0.0.1``, ``PORT`` 参数默认为 ``8000``, 可人为指定：

.. code-block:: shell

   make livehtml AUTOBUILDOPTS="--host 0.0.0.0 --port 1124"

运行上面这个代码将得到类似的实时监控输出：

.. code-block:: shell

   [I 210723 15:35:07 server:335] Serving on http://0.0.0.0:1124
   [I 210723 15:35:07 handlers:62] Start watching changes
   [I 210723 15:35:07 handlers:64] Start detecting changes

.. note::

   背后的原理是：我们使用了 sphinx-autobuild_ 对原有 sphinx-build_ 进行了增强。

.. _sphinx-build: https://www.sphinx-doc.org/en/master/man/sphinx-build.html
.. _sphinx-autobuild: https://github.com/executablebooks/sphinx-autobuild

