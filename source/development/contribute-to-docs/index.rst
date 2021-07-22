.. _contribute-to-docs:

================
如何为文档做贡献
================

.. toctree::
   :hidden:
   :maxdepth: 1
 
   document-style-guide/index
   python-docstring-style-guide
   restructuredtext
   commit-message
   build-the-doc-locally
   translation
   maintainer-responsibility

GitHub 地址：https://github.com/MegEngine/Documentation （欢迎 Star～）

MegEngine 文档的贡献者大致可参考以下几个方向（由易到难）：

* 在 MegEngine/Documentation Issues 或论坛中提供各种各样建设性的意见；
* 更正文档中的拼写错误（Typographical error，简称 Typo）或格式错误（Format error）；
* 更正文档中的其它类型错误，需要详细说明理由，方便进行讨论和采纳；
* 帮助我们进行对 Python API 源码中 Docstring 的翻译，参考 :ref:`translation` ；
* 帮助我们将中文文档内容翻译成对应的英文版本，参考 :ref:`translation` ；
* 帮助我们补充完善 :ref:`文档内容 <doc-content>` ，比如提供大家都需要的教程，不断追求更高的质量。

你也可以浏览处于 Open 状态的 `Issues <https://github.com/MegEngine/Documentation/issues>`_ 
列表，从里面接下任务或找到一些灵感。

.. warning::

   请勿将对文档（Documentation）的 Issues 提交到 MegEngine/MegEngine 存储库。

.. _doc-co-author:

如果你发现了文档中一些很容易改正的细节错误（比如错字、格式不正确等），
但不熟悉 MegEngine 文档的 :ref:`github-collaborate` ，觉得提 Pull Request 太麻烦，
则可以通过 Issues、交流群和论坛等渠道友好地提出来，由我们负责进行后续处理，
并在对应的 Commit 中将你以共同作者（Co-author）的形式加入历史记录：

.. code-block:: shell

   $ git commit -m "Refactor usability tests.
   >
   >
   Co-authored-by: name <name@example.com>
   Co-authored-by: another-name <another-name@example.com>"

这样的方法虽然简单直接，但 MegEngine 团队不能保证处理此类 Issues 的优先级。

俗话说得好：“众人拾柴火焰高”，快来尝试一下吧～ n(\*≧▽≦\*)n

源码组织逻辑
------------

MegEngine 文档的源码结构如下：

.. code-block:: shell
   
   Documentation
   ├── source
   │   ├── _static           # 静态资源文件统一放在这里
   │   ├── _templates
   │   ├── getting-started   #########################
   │   ├── user-guide        #  文档内容主要覆盖区域    #
   │   ├── reference         #  与网页 URL 路径一致    #
   │   ├── development       #########################
   │   ├── conf.py
   │   ├── favicon.ico
   │   ├── logo.png
   │   └── index.rst         
   ├── build                 # 生成的文档一般在 build 目录
   │   ├── doctrees
   │   └── html
   ├── examples
   ├── locales               # Sphinx 多语言支持，内部结构和 source 高度对齐
   │   ├── zh-CN             # 中文：主要需要翻译 API 的 Docstring 部分
   │   └── en                # 英文：需要翻译除 API Docstring 外的全部内容
   ├── Makefile
   ├── requirements.txt
   ├── LICENSE
   ├── CONTRIBUTING.md
   └── README.md

.. note::

   与常见的软件文档 “以英文撰写原文，后续提供多语言翻译” 的逻辑不同，
   MegEngine 文档默认以中文作为原稿，后续提供其它语言的翻译版本。
   但由于 Python Docstring 属于源代码注释的一部分，而代码注释提倡用英文撰写，
   因此 MegEngine 的 Python API 文档将先从源代码提取出英文 Docstring，
   再通过翻译对应的 ``locales/zh-CN/LC_MESSAGES/reference/api/*.po`` 文件变为中文。

.. warning::
   
   为了支持内容的自定义排序，MegEngine 的 API 参考是通过列举而非自动生成的形式添加到文档中的，
   如果你需要在文档中预览新增 API, 则需要手动将他们添加到对应的 ``source/reference/*.rst`` 文件中。

   比如 ``funtional.add`` 位于 ``source/reference/functional.rst`` 的 Arithmetic operations 分类。

   新增 API 不应该出现在当前版本的文档中，所以在验证无误后，请提交到文档的 dev 分支，
   与 MegEngine 的 master 分支对应。（如果更新了对应的使用教程或用户指南，同理。）


如果你曾经使用 Sphinx 构建过 Python 项目的文档，想必会对上面的源码结构非常熟悉。
有的时候，为了在本地预览自己的改动效果，我们需要学会 :ref:`how-to-build-the-doc-locally` 。

.. _doc-content:

内容分类逻辑
------------

MegEngine 的文档主要包括以下方面的内容：

* 教程（Tutorial）：指导读者通过一系列步骤完成项目（或有意义的练习）的课程；
* 指南（Guide）：引导读者完成解决常见问题所需步骤的指南（How-to 系列）；
* 参考（Reference）：用于 API 查询和浏览，方便网络引擎检索的百科全书；
* 解释（Explaination）：对于特定主题、特性的概念性说明。

下面这张表格有助于我们理解它们之间的区别：

+------------------------+------------+----------+
|                        |  学习阶段  | 使用阶段 |
+========================+============+==========+
| 实践步骤               |  教程 🤔   |  指南 📖 |
+------------------------+------------+----------+
| 理论知识               |  解释 📝   |  参考 📚 | 
+------------------------+------------+----------+


更多细节请参考 :ref:`megengine-document-style-guide` 。

.. _github-collaborate:

Git 协作流程
------------

高效清晰的沟通是合作的前提，标准的 MegEngine 文档协作流程如下：

#. 创建一个 Issue，讨论接下来你打算进行什么样的尝试，交流想法；
#. 一旦确定了初步方案，Fork 并 Clone 存储库，创建一个新的本地分支；
#. 修改代码，记录 :ref:`Commit <commit-message>` 并 Push 到你 Fork 的远端分支（Origin）；
#. 在 GitHub 创建一个 Pull Request 并向上游（Upstream）发起合并申请；
#. 根据 Review 意见进行交流，如有需要则修改你的代码；
#. 当你的分支被合并了，便可以删除对应的本地和远程分支。

我们还提供了更加详细的 :ref:`pull-request-guide` （里面的规则适用于 MegEngine ）。

.. _doc-ci-preview:

借助 CI 预览更新后的文档
~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   * 如果你对文档的内容修改较多，建议参考 :ref:`how-to-build-the-doc-locally` 。
   * CI 默认使用 pip 安装的 Wheel 包来构建文档，所以不适用于 :ref:`Python 文档字符串 <python-docstring-style-guide>` 更新。

**当你在本地的修改已经完成后** ，可以通过发起 Pull Request 来触发 GitHub Actions.
借助于 `Build <https://github.com/MegEngine/Documentation/actions/workflows/build.yml>`_ 工作流，
我们可以对整个文档的构建过程进行检查，确保能够正常生成整个文档网站。
当整个工作流成功执行完，你会在 `Actions <https://github.com/MegEngine/Documentation/actions>`_
对应的 workflow runs 结果页面中找到临时生成的构件（Artifacts），
里面是 HTML 形式生成的整个文档网站的压缩包，你可以下载到本地后解压并进行预览。


Pull Request 如何被合并
~~~~~~~~~~~~~~~~~~~~~~~

我们为 ``main`` 分支启用了保护规则，满足以下条件的 Pull Request 才能被成功 Merge:

* 必须签署贡献者协议（Contributor License Agreement，简称 CLA）；
* 必须至少有一位官方维护人员审核（Review）完成并批准（Approve）了你的所有代码
* 必须通过 Actions 中触发的状态检查（Status check），如 
  `Build <https://github.com/MegEngine/Documentation/actions/workflows/build.yml>`_ .
* 必须将你的 Commits 历史记录整理为线性，消息内容符合 `规范 <commit-message>`_.

官网文档页面的更新将会有一定的处理流程延迟，请耐心等待官网服务器更新文档内容。

官方文档维护人员
----------------

当代码需要找人审核时，可以从下面的人员名单中进行选择：

* 架构相关：Chai_
* 主题相关：Chai_ （上游主题 Pydata 相关问题请直接向上游库开发者反馈）
* 教程相关：Chai_
* 文档相关：Chai_

.. _Chai: https://github.com/MegChai
