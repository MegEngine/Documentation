.. _docs:

================
如何为文档做贡献
================

.. toctree::
   :hidden:
   :maxdepth: 1

   document-style-guide/index
   python-docstring-style-guide
   cpp-doc-style-guide
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
* 帮助我们进行对 Python API 源码中文档字符串的翻译，参考 :ref:`translation` ；
* 帮助我们将中文文档内容翻译成对应的英文版本，参考 :ref:`translation` ；
* 帮助我们补充完善 :ref:`文档内容 <doc-content>` ，比如教程和指南，不断追求更高的质量。

你也可以浏览处于 Open 状态的 `Issues <https://github.com/MegEngine/Documentation/issues>`_
列表，从里面接下任务或找到一些灵感。

.. _doc-contribution-quick-start:

非开发人员的贡献方式
--------------------

如果你发现了文档中一些很容易改正的细节错误（比如错字、格式不正确等），
但没有 GitHub 账号，或者不熟悉 MegEngine 文档的 :ref:`github-collaborate` ，
觉得 Fork & Pull Request 等流程过于麻烦，
则可以通过 Issues、交流群和论坛等官方渠道友好地提出来，由我们负责进行后续处理。
我们会在对应的 Commit 中将你以共同作者（Co-author）的形式加入历史记录（需要提供 GitHub 用户名和邮箱）：

.. code-block:: shell

   $ git commit -m "Refactor usability tests.
   >
   >
   Co-authored-by: name <name@example.com>
   Co-authored-by: another-name <another-name@example.com>"

俗话说得好：“众人拾柴火焰高”，有时候发现问题比解决问题还重要，快来尝试一下吧～

.. warning::

   * 这样的方法虽然简单直接，但 MegEngine 团队不能保证处理此类 Issues 的优先级；
   * 请勿将对文档（Documentation）的 Issues 提交到 MegEngine/MegEngine 存储库。

.. _doc-contribution-standard-way:

开发者的贡献方式
----------------

开发环境设置
~~~~~~~~~~~~

.. note::

   目前提供基于 `GitHub Codespaces`_ 云端开发（推荐）和 :ref:`本地开发 <how-to-build-the-doc-locally>` 两种模式。

   如果你的网络环境能够流畅地访问 GitHub 服务，建议在云端进行开发。
   打开 `Documentation <https://github.com/MegEngine/Documentation>`_ 存储库，
   点击 Code -> Codespaces -> New codespace 自动启动实例，完成环境初始化。
   在云端的 Visual Studio Code 中进行相关操作后，执行下面的命令构建文档，启动 WEB 服务进行验证：

   >>> make html
   >>> python3 -m http.server 1124 --directory build/html

   将自动转发的端口可见性设置为公开（Public），可以方便他人在其它机器上进行审核；

   * Codespaces 支持同步 Visual Studio Code 设置或基于 ``dotfiles`` 的 `个性化设置
     <https://docs.github.com/en/codespaces/customizing-your-codespace/personalizing-codespaces-for-your-account>`_ ；
   * Codespaces 支持通过 Visual Studio Code Desktop 打开，开发者可按需选择使用方式。

.. _GitHub Codespaces: https://github.com/features/codespaces

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

.. note::

    与常见的软件文档 “以英文撰写原文，后续提供多语言翻译” 的逻辑不同，
    MegEngine 文档默认以中文作为原稿，后续提供其它语言的翻译版本。
    但由于 :ref:`Python 文档字符串 <python-docstring-style-guide>` 属于源代码注释的一部分，而代码注释提倡用英文撰写，
    因此 MegEngine 中文文档中的 API 页面在自动生成时，将先从源代码提取出英文文档字符串，
    再通根据对应的 ``locales/zh-CN/LC_MESSAGES/reference/api/*.po`` 文件中的翻译替换中文。
    想要了解有关细节，可以参考 :ref:`translation` 页面。

.. warning::

    尽管单个的 API 页面是依据文档字符串的内容自动进行生成的，
    但为了支持 APIs 的自定义排序，MegEngine 的 :ref:`megengine-reference`
    中的各个列表是在 ``source/reference/*.rst`` 文件中人工进行维护的。

    比如 :func:`.functional.add` 位于 :docs:`source/reference/functional.rst` 的 Arithmetic operations 分类。

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

已经推送分支后，可以通过向 Documentation 的特定分支发起 Pull Request 来触发持续集成。

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
   如果你默认使用了非 GitHub 账户，建议使用 ``git config --local`` 命令配置。

.. note::

   官网文档页面的更新将会有一定的处理流程延迟，请耐心等待官网服务器更新文档内容。


