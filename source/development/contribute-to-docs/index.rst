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

.. note::

   GitHub 地址：https://github.com/MegEngine/Documentation （欢迎 Star～）

MegEngine 文档的贡献者大致可参考以下几个方向（由易到难）：

* 在 MegEngine/Documentation `Issues <https://github.com/MegEngine/Documentation/issues>`_ 或
  `论坛 <https://discuss.megengine.org.cn/c/33-category/33>`_ 中提供各种各样建设性的意见；
* 更正文档中的拼写错误（Typographical error，简称 Typo）或格式错误（Format error）；
* 更正文档中的其它类型错误，需要详细说明理由，方便进行讨论和采纳；
* 帮助我们进行对 Python API 源码中 Docstring 的翻译，参考 :ref:`translation` ；
* 帮助我们将中文文档内容翻译成对应的英文版本，参考 :ref:`translation` ；
* 帮助我们补充完善 :ref:`文档内容 <doc-content>` ，比如教程和指南，不断追求更高的质量。

你也可以浏览处于 Open 状态的 `Issues <https://github.com/MegEngine/Documentation/issues>`_ 
列表，从里面接下任务或找到一些灵感。

.. _doc-co-author:

非开发人员的贡献方式
--------------------

如果你发现了文档中一些很容易改正的细节错误（比如错字、格式不正确等），
但没有 GitHub 账号，或者不熟悉 MegEngine 文档的 :ref:`github-collaborate` ，
觉得 Fork & Pull Request 等流程过于麻烦，
则可以通过 Issues、交流群和论坛等官方渠道友好地提出来，由我们负责进行后续处理。
我们会在对应的 Commit 中将你以共同作者（Co-author）的形式加入历史记录：

.. code-block:: shell

   $ git commit -m "Refactor usability tests.
   >
   >
   Co-authored-by: name <name@example.com>
   Co-authored-by: another-name <another-name@example.com>"

这样的方法虽然简单直接，但 MegEngine 团队不能保证处理此类 Issues 的优先级。

俗话说得好：“众人拾柴火焰高”，快来尝试一下吧～ n(\*≧▽≦\*)n

.. warning::

   请勿将对文档（Documentation）的 Issues 提交到 MegEngine/MegEngine 存储库。

开发者的贡献方式
----------------

开发环境设置
~~~~~~~~~~~~

.. note:: 

   * 如果你曾经使用 Sphinx 构建过 Python 项目的文档，想必会对下面展示的源码结构非常熟悉。
     有的时候，为了在本地预览自己的改动效果，我们需要学会 :ref:`how-to-build-the-doc-locally` 。
   * 你也可以根据自身情况，选择使用 `Gitpod <https://gitpod.io/#prebuild/https://github.com/MegEngine/Documentation>`_ 
     等类型的云 IDE 来创建一个临时的文档开发环境，但这需要连接到 GitHub 帐户，且会对你的网络环境有一定的要求。
     另外由于空间限制，将不会安装 MegEngine 包，因此仅支持使用 MINI 模式来生成除 API Reference 外的文档。

源码组织逻辑
~~~~~~~~~~~~

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

.. dropdown:: :fa:`eye, mr-1` 翻译逻辑和新增 API 注意事项

   .. note::

      与常见的软件文档 “以英文撰写原文，后续提供多语言翻译” 的逻辑不同，
      MegEngine 文档默认以中文作为原稿，后续提供其它语言的翻译版本。
      但由于 Python Docstring 属于源代码注释的一部分，而代码注释提倡用英文撰写，
      因此 MegEngine 的 Python API 文档将先从源代码提取出英文 Docstring，
      再通过翻译对应的 ``locales/zh-CN/LC_MESSAGES/reference/api/*.po`` 文件变为中文。
      （参考 :ref:`translation` ）

   .. warning::
      
      为了支持内容的自定义排序，MegEngine 的 :ref:`megengine-reference` 是通过列举而非自动生成的形式添加到文档中的，
      如果你需要在文档中预览 API, 则需要参考已有 API 的分类组织方式，编辑对应的 ``source/reference/*.rst`` 文件。

      比如 ``funtional.add`` 位于 ``source/reference/functional.rst`` 的 Arithmetic operations 分类。

      新增 API 不应该出现在当前版本的文档中，所以在验证无误后，请提交到文档的 dev 分支，
      与 MegEngine 的 master 分支对应。（如果更新了适用于新版本的教程或用户指南，方法同理。）

.. _doc-content:

内容分类逻辑
~~~~~~~~~~~~

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

GitHub 协作流程
~~~~~~~~~~~~~~~

高效清晰的沟通是合作的前提，标准的 MegEngine 文档协作流程如下：

#. 创建一个 Issue，讨论接下来你打算进行什么样的尝试，交流想法；
#. 一旦确定了初步方案，Fork 并 Clone 存储库，创建一个新的本地分支；
#. 参照 :ref:`风格指南 <megengine-document-style-guide>` 和 :ref:`语法规范 <restructuredtext>`
   修改代码或文本，:ref:`在本地构建与预览文档 <how-to-build-the-doc-locally>` ；
#. 按照 :ref:`格式要求 <commit-message>` 记录 Commit 信息（需要使用 GitHub 账号）；
#. 确认所有改动符合预期后，Push 到你 Fork 的远端分支（Origin）；
#. 在 GitHub 创建一个 Pull Request 并向上游（Upstream）发起合并请求；
#. 根据 Review 意见进行交流，根据需求在 Pull Request 中提交后续修改；
#. 当你的分支被合并了，便可以删除对应的本地和远程分支。

我们还提供了更加详细的 :ref:`pull-request-guide` （里面的规则适用于 MegEngine ）。

.. note::

   要求首先创建 Issue 进行讨论的原因是希望避免一些突如其来的 Pull Request 做无用功；
   而具备一定经验的核心开发者可以直接提供一个草稿版本的 Pull Request 直接进行后续的讨论。

.. _doc-ci-preview:

持续集成 & 部署逻辑
~~~~~~~~~~~~~~~~~~~

当你在本地的修改已经完成后，可以通过发起 Pull Request 来触发持续集成。

.. image:: ../../_static/images/megengine-ci.png
   :align: center

.. note::

   * 借助于 `Build <https://github.com/MegEngine/Documentation/actions/workflows/build.yml>`_ 工作流，
     我们可以对整个文档的构建过程进行检查，确保能够正常生成整个文档网站。
   * 如果整个工作流成功执行完，你会在 `Actions <https://github.com/MegEngine/Documentation/actions>`_
     对应的 workflow runs 结果页面中找到临时生成的构件（Artifacts），
     里面是 HTML 形式生成的整个文档网站的压缩包，可以下载到本地后解压预览。
   * 同时，Gitpod Bot 也会为你的 Pull Request 提供一个云端 IDE 环境，并自动完成初始化和预览。
   * Pull Request 被合并到主分支后，将完成后续的部署，在下一次官网更新时能看到你的改动。

Pull Request 如何被合并
~~~~~~~~~~~~~~~~~~~~~~~

我们为 ``main`` 分支启用了保护规则，满足以下条件的 Pull Request 才能被成功合并（Merge）：

* 必须签署贡献者协议（Contributor License Agreement，简称 CLA）；
* 必须至少有一位官方维护人员审核（Review）完成并批准（Approve）了你的所有代码
* 必须通过 Actions 中触发的状态检查（Status check），如 
  `Build <https://github.com/MegEngine/Documentation/actions/workflows/build.yml>`_ .
* 必须将你的 Commits 历史记录整理为线性，消息内容符合 :ref:`commit-message` 。

.. warning::

   签署 CLA 协议要求 commit 中所记录的 Author 账户都是 GitHub 上的账户。
   如果你默认使用了非 GitHub 账户，需要使用 ``git config`` 命令单独配置。

官网文档页面的更新将会有一定的处理流程延迟，请耐心等待官网服务器更新文档内容。

官方文档维护人员
----------------

当代码需要找人审核时，可以从下面的人员名单中进行选择：

* 整体内容：Chai_ , xxr_ , `Dash Chen`_
* 文档架构：Chai_

.. _Chai: https://github.com/MegChai
.. _xxr: https://github.com/xxr3376
.. _Dash chen: https://github.com/dc3671
