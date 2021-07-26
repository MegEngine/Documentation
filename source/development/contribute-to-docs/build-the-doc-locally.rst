.. _how-to-build-the-doc-locally:

如何在本地构建与预览文档
========================

除了 :ref:`doc-ci-preview` 外，在某些时候，我们需要在本地构建与预览文档。

我们以 Ubuntu 18.04 + Python 3.8 环境为例，向你展示从无到有构建 MegEngine 文档的过程。

克隆文档源码到本地
------------------

将存储库克隆到本地（默认为 ``main`` 分支），确保目录下有 ``Makefile`` 文件。

.. code-block:: shell

   git lfs install
   git clone https://github.com/MegEngine/Documentation

.. note::

   为确保正常克隆，上面的命令将安装 LFS_ (Large File Storage) 插件。

.. _LFS: https://git-lfs.github.com/

设置 MegEngine 路径（可选）
---------------------------

.. warning::

   如果你的本地环境并没有 MegEngine 源码，并不会影响整个文档的构建流程。
   但在这个时候会启动 Mini-doc 模式，所有 API Reference 页面将被排除（因为它们依赖源码进行内容生成）。

根据不同的需求，有两种方式将用于构建文档的 MegEngine 导入当前 Python 环境（任选其一即可）：

* 如果你是框架用户，不需要改动 MegEngine 源码，只想在本地完整地构建和预览所有文档的内容，
  或进行简单的增删查改，建议安装最新发布的 MegEngine 稳定版 Wheel 包。
  可以直接使用对应的 ``pip intall`` 命令将已经打包好的 MegEngine 安装到当前的 Python 环境中。
  :ref:`了解如何进行使用 pip 安装。<install>`
* 如果你是研发人员，需要在指定的 MegEngine 分支源代码上生成对应文档，则需要克隆对应分支进行编译构建。
  通过 ``export PYTHONPATH`` 的形式来临时指定特定的 MegEngine 源代码路径，
  这种方式适合开发者需要同时对源码和文档进行维护的情况。:ref:`了解如何进行从源码构建。<install>`

如果你正在维护 MegEngine 的特性分支，需要添加新的 API 到文档中进行预览验证，此时一定需要使用人为编译版本。
并且在文档的 ``source/reference/*.rst`` 将新增 API 添加到合适的位置。 

安装 Sphinx 与 Pydata 主题
--------------------------

MegEngine 文档使用 Sphinx_ 进行整个网站的构建，请运行下面的指令，安装 Sphinx 和相关依赖：

.. _Sphinx: https://www.sphinx-doc.org

.. code-block:: shell

   python3 -m pip install -r requirements.txt

.. dropdown:: :fa:`eye,mr-1` 编辑 Sphinx 文档的配置文件

   通常情况下，你无需对已有配置文件进行任何改动，即可继续进行后面的流程。
   MegEngine 文档对应的 Sphinx 配置文件位于 :docs:`source/conf.py`,
   如需修改请参考官方的 Configuration_ 页面。

   .. _Configuration: https://www.sphinx-doc.org/en/master/usage/configuration.html

   .. note::

      Sphinx 通过 ``conf.py`` 中的 ``import megengine`` 来尝试寻找 MegEngine 包路径，
      你也可以通过其它方式比如人为将 MegEngine 路径添加到 ``sys.path`` 来达到同样的效果。

      * 使用 ``pip`` 安装的路径应该类似于：``/.../lib/.../site-packages/megengine``
      * 从源码编译构建的路径应该类似于： ``/.../MegEngine/imperative/python/megengine``

   .. warning::

      如果你未经过编译，想要直接使用 MegEngine 源码进行文档的构建，
      则将因会缺少编译构建出的动态链接库而无法正常执行 ``import``.

安装文档所用主题
~~~~~~~~~~~~~~~~

接下来我们需要从 MegEngine/pydata-sphinx-theme 克隆 Fork 版 PyData_ 主题：

.. _Pydata: https://github.com/pydata/pydata-sphinx-theme

.. code-block:: shell

   git clone -b dev git@github.com:MegEngine/pydata-sphinx-theme.git

接着安装修改过的主题包：

.. code-block:: shell

   python3 -m pip install --editable pydata-sphinx-theme

安装相关软件包
--------------

Pandoc 转换工具
~~~~~~~~~~~~~~~

nbsphinx_ 是 Sphinx 的一个插件，可以帮助我们对 ``.ipynb`` 格式的 Jupyter Notebook_ 文件进行解析。

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _Notebook: https://jupyter.org/

我们在安装依赖环境时已经安装好了 nbsphinx, 但还需要通过依赖项目 Pandoc_ 来支持转换 Markdown 格式。

.. _Pandoc: https://pandoc.org/

如果你使用的是是 Ubuntu（Debian）操作系统，可以直接使用 ``apt`` 命令进行安装 Pandoc：

.. code-block:: shell

   sudo apt install -y pandoc

如果你使用的是其它操作系统，想要安装 Pandoc，请参考 Pandoc 官方的 `Installing <https://pandoc.org/installing.html>`_ 页面。

Graphviz 绘图工具
~~~~~~~~~~~~~~~~~

Graphviz_ 是非常流行的图形可视化软件，在 MegEngine 文档中经常会用他制作一些可视化图片。

如果你使用的是是 Ubuntu（Debian）操作系统，可以直接使用 ``apt`` 命令进行安装 Pandoc：

.. code-block:: shell

   sudo apt install -y graphviz 

如果你使用的是其它操作系统，想要安装 Graphviz，请参考 Graphviz 官方的 `Download <https://graphviz.org/download/>`_ 页面。

.. _Graphviz: https://graphviz.org/

使用 Sphinx 进行文档构建
------------------------


在文档目录下使用 ``make html`` 指令，会在 ``build`` 目录下生成 HTML 文件夹。

文档生成成功后，打开 ``build/html/index.html`` 文件便可访问主页。

.. note::

   Sphinx 默认支持增量构建，当你再次执行 ``make html`` 时将仅对变化的文件进行更新；

.. warning::

   Sphinx 不会检测增量模式下非文档文件的更改，例如主题文件、静态文件和与 autodoc 一起使用的源代码；
   如果发现一些页面的元素仍被缓存而没有被更新，请尝试通过传入 ``-a`` 参数禁用增量模式（但构建速度会相应地变慢），
   或者通过 ``make clean`` 指令清除掉已经构建出的内容。

.. note::

   * 运行 ``make help`` 指令，可看到相应的帮助和参数信息，比如显示当前 MegEngine 路径等；
   * ``make html`` 本质上调用了 sphinx-build_ 工具（参考 :docs:`Makefile` 了解更多细节）。 

自动构建和实时预览页面
----------------------

你也可以使用 ``make livehtml`` 指令，在监测到文件变化时自动重新构建，
而且可以通过浏览器进行实时的预览。
``HOST`` 默认为 ``127.0.0.1`` 和 ``PORT`` 默认为 ``8000``, 可人为指定：

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

