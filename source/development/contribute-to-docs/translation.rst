.. _translation:

====================
如何帮助翻译文档内容
====================

MegEngine 文档使用 Sphinx 官方推荐的 `国际化 <build-the-doc-locally>`_ 方式实现多语言支持。

目录结构
--------

对于参与文档翻译的人员，对以下目录结构有一定了解将有所帮助：

.. code-block:: shell

   Documentation
   ├── source                
   ├── locales               # Sphinx 多语言支持，内部结构和 source 高度对齐
   │   ├── gettext           # 原始消息：提取 rst 文件所生成的模版目录
   │   ├── zh-CN             # 中文：主要需要翻译 API 的 Docstring 部分
   │   └── en                # 英文：需要翻译除 API Docstring 外的全部内容
   ...

基本原理
--------

整个翻译内容的生成流程如下：

#. 在 ``source`` 文件夹中存放着所有的文档内容，以 ``.rst`` 格式提供。
#. 通过运行下面的命令，将从 ``.rst`` 文件中提取出可被翻译的消息（message）：

   .. code-block:: shell

      make gettext BUILDDIR=locales

   生成的 ``.pot`` 文件将被放在 ``locales/gettext`` 目录内。

#. 根据需要支持的语言，生成对应的 ``.po`` 文件中提取出可被翻译的消息（message）：

   .. code-block:: shell

      sphinx-intl update -p locales/gettext -l zh-CN -l en

   上面的代码将为我们生成中文和英文两个版本的 ``.po`` 文件，分别位于：

   * ``/locales/zh-CN/LC_MESSAGES/``
   * ``/locales/en/LC_MESSAGES/``

#. 翻译人员需要做的就是翻译 ``.po`` 文件中的内容。样例如下：

   .. code-block:: shell

      #: locales/zh-CN/LC_MESSAGES/example.rst:4
      msgid "Welcome to use MegEngine."
      msgstr "欢迎使用天元 MegEngine."

   另一种情况是，msgid 是多行文本，包含 reStructuredText 语法：

   .. code-block:: shell

      #: ../../builders.rst:9
      msgid ""
      "These are the built-in Sphinx builders. More builders can be added by "
      ":ref:`extensions <extensions>`."
      msgstr ""
      "FILL HERE BY TARGET LANGUAGE FILL HERE BY TARGET LANGUAGE FILL HERE "
      "BY TARGET LANGUAGE :ref:`EXTENSIONS <extensions>` FILL HERE."

   .. warning::

      请注意不要破坏 reST 语法，参考 :ref:`restructuredtext` 。

#. 生成翻译后的文档（ ``LANGUAGE`` 参数默认为 ``zh-CN`` ）：

   .. code-block::

      make LANGUAGE="zh-CN" html

.. note::

   当用户在 GitHub 发起一个翻译相关的 Pull Request 时，
   每次提交新的 commit 会自动触发临时文档的生成和更新。
   但我们推荐用户首先 :ref:`在本地构建文档 <how-to-build-the-doc-locally>` 以确保格式和内容正确。

主要翻译需求
------------

与常见的软件文档 “以英文撰写原文，后续提供多语言翻译” 的逻辑不同，
MegEngine 文档默认以中文作为原稿，后续提供其它语言的翻译版本。
但由于 Python Docstring 属于源代码注释的一部分，而代码注释提倡用英文撰写，
因此 MegEngine 的 Python API 文档将先从源代码提取出英文 Docstring，
再通过翻译对应的 ``locales/zh-CN/LC_MESSAGES/reference/api/`` 中的 ``.po`` 文件变为中文。

官方翻译未必准确、完备，欢迎大家帮助天元 MegEngine 改进文档翻译～

基本的 :ref:`commit message <commit-message>` 形式如下：

.. code-block:: shell

   trans(funtional): add docstring translation for megengine.funtional.add

如果是对已有翻译内容进行了修改，请在 commit message 中详细说明修改原因。

