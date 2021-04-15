��    0      �                5        S  O   d  (   �  R   �  7   0     h  Z   �  �  �  \   �  �     Z   �  h        y  W   �  V   �  T   ;	  V   �	     �	  �   
  �   �       u   �  _     �   n  ^   �    Q  Q   `  N   �  d     N   f  j   �  \      j   }  h   �  T   Q     �  Y   �       `   *  i   �     �  G   
     R  Z   e  g   �  Y   (  �  �  5        B  O   S  (   �  R   �  7        W  Z   t  �  �  \   �  �     Z   �  h   �     h  W   {  V   �  T   *  V        �  �  �  �   �     n   u   �   _   �   �   ]!  ^   �!    @"  Q   O#  N   �#  d   �#  N   U$  j   �$  \   %  j   l%  h   �%  T   @&     �&  Y   �&     '  `   '  i   z'     �'  G   �'     A(  Z   T(  g   �(  Y   )   **请不要突然直接发起一个 Pull Request!!!** Git 协作流程 GitHub 地址：https://github.com/MegEngine/Documentation （欢迎 Star～） MegEngine 文档的源码结构如下： MegEngine 文档的贡献者大致可参考以下几个方向（由易到难）： MegEngine 的文档主要包括以下方面的内容： Pull Request 如何被合并 一旦确定了初步方案，Fork 并 Clone 存储库，创建一个新的本地分支； 与常见的软件文档 “以英文撰写原文，后续提供多语言翻译” 的逻辑不同， MegEngine 文档默认以中文作为原稿，后续提供其它语言的翻译版本。 但由于 Python Docstring 属于源代码注释的一部分，而代码注释提倡用英文撰写， 因此 MegEngine 的 Python API 文档将先从源代码提取出英文 Docstring， 再通过翻译对应的 ``locales/zh-CN/LC_MESSAGES/reference/api/*.po`` 文件变为中文。 主题相关：Chai_ （上游主题 Pydata 的研发请直接向上游负责人员反馈） 你也可以浏览处于 Open 状态的 `Issues <https://github.com/MegEngine/Documentation/issues>`_ 列表，从里面接下任务或找到一些灵感。 俗话说得好：“众人拾柴火焰高”，快来尝试一下吧～ n(\*≧▽≦\*)n 修改代码，:ref:`提交 Commit <commit-message>` 并 Push 到你 Fork 的远端分支（Origin）； 内容分类逻辑 创建一个 Issue，讨论接下来你打算进行什么样的尝试，交流想法； 参考（Reference）：用于 API 查询和浏览，方便网络引擎检索的字典 在 GitHub 创建一个 Pull Request 并向上游（Upstream）申请将其合并； 在 MegEngine/Documentation Issues 或论坛中提供各种各样建设性的意见； 如何为文档做贡献 如果你发现了文档中一些很容易改正的细节错误（比如错字、格式不正确等）， 但不熟悉 MegEngine 文档的 :ref:`github-collaborate` ，觉得提 Pull Request 太麻烦， 则可以通过 Issues、交流群和论坛等渠道友好地提出来，由我们负责进行后续处理， 并在对应的 Commit 中将你以共同作者（Co-author）的形式加入历史记录： 如果你曾经使用 Sphinx 构建过 Python 项目的文档，想必会对上面的源码结构非常熟悉。 有的时候，为了在本地预览自己的改动效果，我们需要学会 :ref:`how-to-build-the-doc-locally` 。 官方文档维护人员 官网文档页面的更新将会有一定的处理流程延迟，请耐心等待官网服务器更新文档内容。 帮助我们将中文文档内容翻译成对应的英文版本，参考 :ref:`translation` ； 帮助我们补充完善 :ref:`文档内容 <doc-content>` ，比如提供大家都需要的教程，不断追求更高的质量。 帮助我们进行对 Python API 源码中 Docstring 的翻译，参考 :ref:`translation` ； 当 Pull Request 被成功合入 ``main`` 分支，将自动触发 ``gh-pages`` 分支上的部署流程。 ``gh-pages`` 分支将被用于 GitHub Pages 静态网站服务的部署， 可作为 MegEngine 官网由于不可抗力等因素无法访问时的镜像站点。 当代码需要找人审核时，可以从下面的人员名单中进行选择： 当你的分支被合并了，便可以删除对应的本地和远程分支。 必须将你的 Commits 历史记录整理为线性，消息内容符合 `规范 <commit-message>`_. 必须签署贡献者协议（Contributor License Agreement，简称 CLA）； 必须至少有一位官方维护人员审核（Review）完成并批准（Approve）了你的所有代码 必须通过 Actions 中触发的状态检查（Status check），如 Sphinx Build and Test. 我们为 ``main`` 分支启用了保护规则，满足以下条件的 Pull Request 才能被成功 Merge: 我们还提供了更加详细的 :ref:`pull-request-guide` （里面的规则适用于 MegEngine ）。 指南（Guide）：针对特定情景/特定功能进行详细说明的操作手册 教程相关：Chai_ 教程（Tutorial）：可从零开始完整跑通的教学性材料，适合快速上手 文档相关：Chai_ 更正文档中的其它类型错误，需要详细说明理由，方便进行讨论和采纳； 更正文档中的拼写错误（Typographical error，简称 Typo）或格式错误（Format error）； 架构相关：Chai_ 根据 Review 意见进行交流，如有需要则修改你的代码； 源码组织逻辑 请勿将对文档（Documentation）的 Issues 提交到 MegEngine/MegEngine 存储库。 这样的方法虽然简单直接，但 MegEngine 团队不能保证处理此类 Issues 的优先级。 高效清晰的沟通是合作的前提，标准的 MegEngine 文档协作流程如下： Project-Id-Version: MegEngine 1.3.0
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-14 14:17+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_Hans_CN
Language-Team: zh_Hans_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 **请不要突然直接发起一个 Pull Request!!!** Git 协作流程 GitHub 地址：https://github.com/MegEngine/Documentation （欢迎 Star～） MegEngine 文档的源码结构如下： MegEngine 文档的贡献者大致可参考以下几个方向（由易到难）： MegEngine 的文档主要包括以下方面的内容： Pull Request 如何被合并 一旦确定了初步方案，Fork 并 Clone 存储库，创建一个新的本地分支； 与常见的软件文档 “以英文撰写原文，后续提供多语言翻译” 的逻辑不同， MegEngine 文档默认以中文作为原稿，后续提供其它语言的翻译版本。 但由于 Python Docstring 属于源代码注释的一部分，而代码注释提倡用英文撰写， 因此 MegEngine 的 Python API 文档将先从源代码提取出英文 Docstring， 再通过翻译对应的 ``locales/zh-CN/LC_MESSAGES/reference/api/*.po`` 文件变为中文。 主题相关：Chai_ （上游主题 Pydata 的研发请直接向上游负责人员反馈） 你也可以浏览处于 Open 状态的 `Issues <https://github.com/MegEngine/Documentation/issues>`_ 列表，从里面接下任务或找到一些灵感。 俗话说得好：“众人拾柴火焰高”，快来尝试一下吧～ n(\*≧▽≦\*)n 修改代码，:ref:`提交 Commit <commit-message>` 并 Push 到你 Fork 的远端分支（Origin）； 内容分类逻辑 创建一个 Issue，讨论接下来你打算进行什么样的尝试，交流想法； 参考（Reference）：用于 API 查询和浏览，方便网络引擎检索的字典 在 GitHub 创建一个 Pull Request 并向上游（Upstream）申请将其合并； 在 MegEngine/Documentation Issues 或论坛中提供各种各样建设性的意见； 如何为文档做贡献 如果你发现了文档中一些很容易改正的细节错误（比如错字、格式不正确等）， 但不熟悉 MegEngine 文档的 :ref:`github-collaborate` ，觉得提 Pull Request 太麻烦， 则可以通过 Issues、交流群和论坛等渠道友好地提出来，由我们负责进行后续处理， 并在对应的 Commit 中将你以共同作者（Co-author）的形式加入历史记录： 如果你曾经使用 Sphinx 构建过 Python 项目的文档，想必会对上面的源码结构非常熟悉。 有的时候，为了在本地预览自己的改动效果，我们需要学会 :ref:`how-to-build-the-doc-locally` 。 官方文档维护人员 官网文档页面的更新将会有一定的处理流程延迟，请耐心等待官网服务器更新文档内容。 帮助我们将中文文档内容翻译成对应的英文版本，参考 :ref:`translation` ； 帮助我们补充完善 :ref:`文档内容 <doc-content>` ，比如提供大家都需要的教程，不断追求更高的质量。 帮助我们进行对 Python API 源码中 Docstring 的翻译，参考 :ref:`translation` ； 当 Pull Request 被成功合入 ``main`` 分支，将自动触发 ``gh-pages`` 分支上的部署流程。 ``gh-pages`` 分支将被用于 GitHub Pages 静态网站服务的部署， 可作为 MegEngine 官网由于不可抗力等因素无法访问时的镜像站点。 当代码需要找人审核时，可以从下面的人员名单中进行选择： 当你的分支被合并了，便可以删除对应的本地和远程分支。 必须将你的 Commits 历史记录整理为线性，消息内容符合 `规范 <commit-message>`_. 必须签署贡献者协议（Contributor License Agreement，简称 CLA）； 必须至少有一位官方维护人员审核（Review）完成并批准（Approve）了你的所有代码 必须通过 Actions 中触发的状态检查（Status check），如 Sphinx Build and Test. 我们为 ``main`` 分支启用了保护规则，满足以下条件的 Pull Request 才能被成功 Merge: 我们还提供了更加详细的 :ref:`pull-request-guide` （里面的规则适用于 MegEngine ）。 指南（Guide）：针对特定情景/特定功能进行详细说明的操作手册 教程相关：Chai_ 教程（Tutorial）：可从零开始完整跑通的教学性材料，适合快速上手 文档相关：Chai_ 更正文档中的其它类型错误，需要详细说明理由，方便进行讨论和采纳； 更正文档中的拼写错误（Typographical error，简称 Typo）或格式错误（Format error）； 架构相关：Chai_ 根据 Review 意见进行交流，如有需要则修改你的代码； 源码组织逻辑 请勿将对文档（Documentation）的 Issues 提交到 MegEngine/MegEngine 存储库。 这样的方法虽然简单直接，但 MegEngine 团队不能保证处理此类 Issues 的优先级。 高效清晰的沟通是合作的前提，标准的 MegEngine 文档协作流程如下： 