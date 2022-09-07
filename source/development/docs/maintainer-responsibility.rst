.. _maintainer-responsibility:

==========================
MegEngine 文档维护人员职责
==========================

MegEngine 文档的多分支维护逻辑与 MegEngine 源码的维护逻辑不同：

* 当某个 MegEngine 版本正式对外发布时，对应版本的文档也会发布在文档官网；
* ``main`` 分支用于在当前已经发布的 ``stable`` 版本做内容的补充或修复；
* ``dev`` 分支用于在未发布版本，或者是 ``rc`` 版本做内容的提前更新。
* 在新的 ``stable`` 版本发布后，通常会将 ``dev`` 分支的变动 ``rebase`` 到 ``main`` 分支。

.. note:: 

   * 文档不区分像 1.6.0 和 1.6.1 这种 1.6.x 版本号的区别，统一看作是 1.6 版本文档；
   * ``dev`` 分支需要定期做 ``rebase main`` 操作，使之实际上是下一个正式版的预发布版本。

.. warning:: 

   ``dev`` 分支不使用已发布的 Wheel 包，而是每次都从 GitHub 上 MegEnine 的源码进行最简编译构建，
   因此其 CI 用时将远超过日常维护的 ``main`` 分支（约 1 小时），通常更新频率极低，且内容较为集中。

基本要求
--------

* 知道 :ref:`how-to-build-the-doc-locally` , 熟悉 :ref:`restructuredtext` 中内容；
* 知道 :ref:`translation`, 与项目管理人员进行相关工作量的协调沟通工作；
* 知道 :ref:`cpp-doc-style-guide`, 能够更新相关的 C++ 文档；
* 具备技术写作经验，完全理解 :ref:`megengine-document-style-guide` 的含义。

请假设你在维护一份不断更新的出版物，时刻思考内容组织的合理性，小心破窗效应。

版本发布
--------

.. seealso::

   在 :docs:`scripts/bump_version.sh` 提供了一份参考执行脚本。

每当 MegEngine 的新版本成功发布时（即 GitHub Released 并且 CDN 上的 Wheel 包为最新版），
文档的维护者需要完成以下流程（假定当前新发布版本为 v1.11）：

GitHub 存储库变动
~~~~~~~~~~~~~~~~~~~

请确认已经在本地安装好最新的 MegEngine Wheel 包，再执行如下操作：

#. 将 Documentation 的 ``main`` 分支分叉出 ``release/v1.10`` 分支并 ``push`` 到 GitHub 作为备份；
#. 将 Documentation 的 ``dev`` 分支上的改动 ``rebase`` 到 ``main`` 分支后合并，删除旧的 ``dev`` 分支；
#. *版本相关。* 在 ``main`` 分支上进行如下修改：

   * 修改 ``README.md`` 中对分支名的解释，对 ``main`` 和 ``dev`` 分支名含义进行更新；
   * 修改 ``source/conf.py`` 中的 ``version`` 变量值，在本例中为 ``"1.10"`` 改为 ``"1.11"``;
   * 修改 ``requirements.txt`` 中的第一行 Cache key, 使 CI 获取最新版本的 Wheel 包；
   * 记录并推送更改，建议 ``commit`` 消息为内容 ``chore: bump version`` 之类
#. *翻译相关。* 在 ``main`` 分支上进行如下修改：

   * 执行 ``make gettext`` 提取出可被翻译的消息模版；
   * 执行 ``sphinx-intl update -p build/gettext -l zh_CN -l en`` 更新可被翻译的条目；
   * 记录并推送更改，建议 ``commit`` 消息为内容 ``trans: update po files`` 之类
#. 从 ``main`` 分支分叉出新的 ``dev`` 分支，该分支负责为 v1.12 版本的改动提前进行维护；

完成上述流程后，Documentation 的 GitHub 存储库上所需做的修改已经完成。

更新 JSON 文件配置*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

想要使用户能够访问到不同版本（包括新发布）的 MegEngine 文档，还需要对两个 JSON 文件进行维护：

* ``version.json``: 用于为文档主题中的版本切换器的下拉选项进行设置，
  `可直接访问 <https://www.megengine.org.cn/doc/version.json>`_ ；
* ``mapping.json``: 对 OSS 上部署的文档实例与官网线上所部署的版本进行映射。

.. admonition:: 相关文件更新流程已经自动化

   ``mapping.json`` 和 ``version.json`` 的内容更新已经能够由 CI 
   （用到了 :docs:`scripts/oss/update.py` 和 :docs:`scripts/oss/gen_version.py` 脚本） 自动地更新，默认无需额外关注。
   二者依赖一些约定来确保行为的正常：

   * ``mapping.json`` 会在每个 commit 写入 ``source/conf.py`` 中的版本号，
     因此随着仓库的推移， ``mapping.json`` 中会逐渐存下全部历史的 ``v1.x`` 的最后一个文档 commit ID;
   * ``mapping.json`` 中还会额外保存 main 分支写入的 ``stable`` key 和 dev 分支写入的 ``master`` key;
   * ``version.json`` 生成时，会要求 ``mapping.json`` 必须只包含上述的 key，随后提取出所有的 key 生成对应的结构。

   针对每一个 commit id, 将通过 :docs:`scripts/oss/build_for_oss.sh`
   与 CI 生成符合要求的 ``static.tgz`` 文件，并同步到 OSS 的相应 Bucket 中。
   如需要调整上述逻辑，开发人员需要去 OSS 中修改对应的文件，确保文件是符合修改后的代码逻辑的。

.. seealso::

   * 详细说明请参考内部 Wiki 中的《官网架构设计》有关内容，
     并搞懂目前的 `CI 部署 <https://github.com/MegEngine/Documentation/blob/main/.github/workflows/deploy.yml>`_ 逻辑；
   * 想要了解 ``version.json`` 的设计初衷，请阅读 :ref:`megengine-doc-theme` 。

.. warning::

   ``mapping.json`` 通过 ``source/conf.py`` 中的 ``version`` 变量来获取每次更新映射时所用到的 key 值，
   因此请确保该值的绝对正确性，否则可能导致一些意想不到的后果（如将已归档的文档进行了更新）。

.. _megengine-doc-theme:

MegEngine 文档主题
-----------------------------------

MegEngine 使用了 Fork 版本的 Pydata Sphinx 主题（注意是 ``dev`` 分支）：

https://github.com/MegEngine/pydata-sphinx-theme/tree/dev

在原有主题的基础设计上支持了语言和版本切换功能。
