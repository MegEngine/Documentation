.. _how-to-build-the-doc-locally:

如何在本地构建文档
==================

克隆文档源码到本地
------------------

将存储库克隆到本地（默认为 ``main`` 分支），确保目录下有 ``Makefile`` 文件。

.. code-block:: shell

   git clone https://github.com/MegEngine/Documentation

.. note::

   为确保正常克隆，请确保本地 Git 已经安装 LFS_ (Large File Storage) 插件。

.. _LFS: https://git-lfs.github.com/

设置 MegEngine 路径
-------------------

根据不同的需求，有两种方式将用于构建文档的 MegEngine 导入当前 Python 环境（任选其一即可）：

* 如果你是框架用户，不需要改动 MegEngine 源码，只需在本地构建和预览文档的内容，
  或对进行简单的增删查改，建议安装最新发布的 MegEngine 稳定版 Wheel 包构建文档。
  可以直接使用对应的 ``pip intall`` 命令将已经打包好的 MegEngine 安装到当前的 Python 环境中。
* 如果你是研发人员，需要在指定的 MegEngine 分支源代码上生成对应文档，则需要克隆对应分支进行编译构建。
  通过 ``export PYTHONPATH`` 的形式来临时指定特定的 MegEngine 源代码路径，
  这种方式适合开发者需要同时对源码和文档进行维护的情况。:ref:`了解如何进行从源码构建。<install>` 

.. warning::
   
   为了支持内容的自定义排序，MegEngine 的 API 参考是通过列举而非自动生成的形式添加到文档中的，
   如果你需要在文档中预览新增 API, 则需要手动将他们添加到对应的 ``source/reference/*.rst`` 文件中。

   比如 ``funtional.add`` 位于 ``source/reference/functional.rst`` 的 Arithmetic operations 分类。

   新增 API 不应该出现在当前版本的文档中，所以在验证无误后，请提交到文档的 dev 分支，
   与 MegEngine 的 master 分支对应。（如果更新了对应的使用教程或用户指南，同理。）

安装 Sphinx 与 Pydata 主题
--------------------------

MegEngine 文档使用 Sphinx_ 进行整个网站的构建，请运行下面的指令，安装 Sphinx 和相关依赖：

.. _Sphinx: https://www.sphinx-doc.org

.. code-block:: shell

   python3 -m pip install -r requirements.txt

MegEngine 文档对应的 Sphinx 配置文件位于 ``source/conf.py`` ，如需修改请参考官方的 Configuration_ 页面。

.. _Configuration: https://www.sphinx-doc.org/en/master/usage/configuration.html

通常情况下，你无需对已有配置文件进行任何改动，即可继续进行后面的流程。

.. note::

   Sphinx 在应用配置时将通过执行上面脚本中的 ``import megengine`` 来尝试寻找 MegEngine 包路径。

   * 使用 ``pip`` 安装的路径应该类似于：``/.../lib/.../site-packages/megengine``
   * 从源码编译构建的路径应该类似于： ``/.../MegEngine/imperative/python/megengine``

接下来我们需要从 MegEngine/pydata-sphinx-theme 安装 Fork 版 PyData_ 主题：

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

在文档目录下使用 ``make help`` 指令，可看到相应的帮助信息。

在文档目录下使用 ``make html`` 指令，可根据 ``BUILDDIR`` 路径（默认为 ``build`` ）生成 HTML 文件夹。

.. note::

   * Sphinx 支持增量构建，当你对源文件进行了更改并保存，只需再次执行 ``make html`` 即可。
   * **如果发现一些页面的元素仍被缓存而没有被更新** ，请尝试先执行 ``make clean`` 指令。
   * 本质上所有的指令通过 sphinx-build_ 执行，阅读 ``Makefile`` 文件源代码，可以了解更多细节。

.. _sphinx-build: https://www.sphinx-doc.org/en/master/man/sphinx-build.html

文档生成成功后，打开 ``build/html/index.html`` 文件便可访问主页。

启动本地 Web 服务器（可选）
---------------------------

如果你有在本地启动 Web 服务器的需求，一种比较简单的方法是使用 Python 自带的 ``http`` 模块：

.. code-block:: shell

   python3 -m http.server 1124 --directory build/html

运行上面的代码，可将本地的 build/html 下的 Web 服务映射到 1124 端口，你也可以选择使用其它 Web 服务器。

如果你的 Python 版本低于 3.7, 将不支持 ``--directory`` 参数，请 ``cd`` 到对应目录执行上述命令。

通常你可以选择将 Web 服务挂在后台，这样在重新 build HTML 文件后只需要刷新页面即可。

