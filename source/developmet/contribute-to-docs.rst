.. _contribute-to-docs:

================
如何为文档做贡献
================
目前文档所用存储库地址为：https://github.com/MegEngine/Tutorials （将来会进行改动）

.. toctree::
   :hidden:
   :maxdepth: 1
   
   restructuredtext.rst


文档的组织逻辑
--------------
内容正在建设中...

在本地构建文档
--------------

克隆文档源码到本地
~~~~~~~~~~~~~~~~~~
将存储库克隆到本地（默认为 ``main`` 分支），确保目录下有 `Makefile` 文件。

.. code-block:: shell

   git clone --depth=1 https://github.com/MegEngine/Tutorials

.. note::

   为确保正常克隆，请确保本地已经安装 Git LFS_ (Large File Storage) 插件。

.. _LFS: https://git-lfs.github.com/

设置 MegEngine 路径
~~~~~~~~~~~~~~~~~~~
根据不同的需求，有两种方式将用于构建文档的 MegEngine 导入当前 Python 环境（任选其一即可）：

   - 如果你不需要改动 MegEngine 源码，只需在本地预览，或对文档内容进行增删查改，
     建议安装最新发布的 MegEngine 稳定版Wheel 包构建文档。
     可以直接使用 `python3 -m pip intall megengine` 命令
     将已经打包好的 MegEngine 安装到当前的 Python 环境中。
   - 如果你需要在指定的 MegEngine 分支上生成对应文档，则需要克隆对应的 MegEngine 分支进行编译构建。
     这种方式适合开发者需要同时对文档进行维护的情况。:ref:`了解如何进行从源码构建。<install>` 

安装 Sphinx 与依赖项
~~~~~~~~~~~~~~~~~~~~
MegEngine 文档使用 Sphinx_ 进行整个网站的构建，请运行下面的指令，确保安装了 Sphinx 和相关依赖：

.. _Sphinx: https://www.sphinx-doc.org

.. code-block:: shell

   python3 -m pip install -r requirements.txt

MegEngine 文档的对应 Sphinx 配置文件位于 ``source/conf.py`` ，如需修改请参考官方的 Configuration_ 页面。

.. _Configuration: https://www.sphinx-doc.org/en/master/usage/configuration.html

通常情况下，你无需对已有配置文件进行任何改动，即可继续进行后面的流程。

.. note::

   Sphinx 在应用配置时将通过执行上面脚本中的 ``import megengine`` 来尝试寻找 MegEngine 包路径。
   使用 `make info` 指令，可以看到当前的 ``MegEngine`` 路径和构建参数等信息。

      - 从 ``pip`` 安装的路径应该类似于：``/.../lib/.../site-packages/megengine``
      - 从源码编译构建的路径应该类似于： ``/.../MegEngine/imperative/python/megengine``

接下来我们需要从 MegEngine/pydata-sphinx-theme （一个 Fork 版本）安装 PyData_ 主题：

.. _Pydata: https://github.com/pydata/pydata-sphinx-theme

.. code-block:: shell

   git clone -b dev --depth=1 git@github.com:MegEngine/pydata-sphinx-theme.git

接着进入该目录，安装修改过的主题包：

.. code-block:: shell

   python3 -m pip install --editable .

安装 Pandoc
~~~~~~~~~~~
nbsphinx_ 是 Sphinx 的一个插件，可以帮助我们对 ``*.ipynb`` 格式文件进行解析并生成对应页面。

我们在上一小节已经安装好了 nbsphinx, 但 nbsphinx 还需要通过依赖项目 Pandoc_ 来支持转换 Markdown 格式。

如果你使用的是是 Ubuntu（Debian）操作系统，可以直接使用 ``apt`` 命令进行安装 Pandoc：

.. code-block:: shell

   sudo apt install -y pandoc

如果你使用的是其它操作系统，想要安装 Pandoc，请参考 Pandoc 官方的 `Installing`_ 页面。

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _Pandoc: https://pandoc.org/
.. _Installing: https://pandoc.org/installing.html

使用 Sphinx 进行文档构建
~~~~~~~~~~~~~~~~~~~~~~~~

在文档目录下使用 `make html` 指令，可根据 ``BUILDDIR`` 路径（默认为 ``build`` ）生成 html 目录。

在文档目录下使用 `make help` 指令，可看到对应的帮助信息。

.. note::

   Sphinx 支持增量构建，当你对源文件进行了更改并保存，只需再次执行 `make html` 即可。

   如果发现一些页面的元素仍被缓存而没有被更新，请尝试先执行 `make clean` 指令。

维护人员须知
------------

作为文档的维护人员，需要熟练掌握 reStructuredText_ 的基本语法。

.. _reStructuredText: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html
