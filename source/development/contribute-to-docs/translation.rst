.. _translation:

====================
如何帮助翻译文档内容
====================

.. note::

  MegEngine 文档翻译工作目前通过 :ref:`Crowdin <crowdin-tr>` 平台进行协作，与 GitHub 自动集成。

对于参与文档翻译的人员，对以下目录结构有一定了解将有所帮助：

.. code-block:: shell

   Documentation
   ├── source                
   ├── locales               # Sphinx 多语言支持，内部结构和 source 高度对齐
   │   ├── zh-CN             # 中文：主要需要翻译 API 的 Docstring 部分
   │   └── en                # 英文：需要翻译除 API Docstring 外的全部内容
   ...

基本原理
--------

MegEngine 文档使用 Sphinx 官方推荐的 
`国际化 <https://www.sphinx-doc.org/en/master/usage/advanced/intl.html>`_ 方式实现多语言支持。

整个翻译内容的生成流程如下（ **翻译人员通常只需要关注第 4 步** ）：

#. 在 ``source`` 文件夹中存放着所有的文档内容，以 ``.rst`` 格式提供。
#. 通过运行下面的命令，将从 ``.rst`` 文件中提取出可被翻译的消息（message）模版：

   .. code-block:: shell

      make gettext

   生成的 ``.pot`` 文件将被放在 ``build/gettext`` 目录内。

#. 根据需要支持的语言，生成对应的 ``.po`` 文件：

   .. code-block:: shell

      sphinx-intl update -p build/gettext -l zh_CN -l en

   上面的代码将为我们生成中文和英文两个版本的 ``.po`` 文件，分别位于：

   * ``/locales/zh_CN/LC_MESSAGES/``
   * ``/locales/en/LC_MESSAGES/``

#. 翻译人员需要做的就是翻译 ``.po`` 文件中的内容。样例如下：

   .. code-block:: po

      #: locales/zh_CN/LC_MESSAGES/example.rst:37
      msgid "Welcome to use MegEngine."
      msgstr "欢迎使用 MegEngine."

#. 生成翻译后的文档（ ``LANGUAGE`` 参数默认为 ``zh-CN`` ）：

   .. code-block:: shell

      make LANGUAGE="zh_CN" html  # 中文
      make LANGUAGE="en" html     # 英文

现在你可以通过下面提到的 Crowdin 平台来完成对 ``.po`` 文件的翻译，而不需要通过 GitHub.

.. _crowdin-tr:

使用 Crowdin 进行翻译
---------------------

传统的软件翻译流程通常需要面临审核校对和信息同步等难题，
通过 Crowdin 则可以享受到：

* 开放参与。任何人都可以注册账号，加入团队，翻译贡献。
  每个人的翻译记录都会被记录下来，而不会被覆盖或丢弃。
  贡献者之间相互审核建议，选择最优的翻译。
* 透明运作。任何操作都会记录在活动日志当中，有不当翻译或者蓄意破坏很容易被发现并撤销。
  翻译人员可以相互讨论提问，提供单独的讨论板块。
* 进度清晰。当有新的内容需要翻译时，或对原始内容进行修改后，能在第一时间被监控到。
* 记忆系统。可以根据已有翻译内容给出翻译建议，也可使用主流神经翻译接口辅助建议。
* 自动集成。更新的翻译内容将自动通过集成系统同步到 GitHub 特定分支，再由人工进行合入。

更多特性请参考：https://crowdin.com/features

加入 MegEngine 翻译项目
~~~~~~~~~~~~~~~~~~~~~~~

你需要注册一个 `Crowdin <https://crowdin.com/>`_ 账户，
进入 `MegEngine 项目页面 <https://crowdin.com/project/megengine>`_ 后可以看到语言选项卡和整体进度：

Chinese Simplified
  由于 Sphinx 生成网站时使用 Python Docstring 提取 API 文档内容，
  源代码均为英文，因此这部分内容依旧为英文，我们需要将其翻译成中文。

English （以及其它语言）
  与上情况相反：除 Python Docstring 外，所有文档原文内容均为中文，
  因此我们需要将这些内容翻译成指定语言。（通常我们会提供机器翻译版本作为参考）

选择语言后，可以看到多个需要翻译的文件。每个翻译文件和文件夹都有一个翻译进度。
蓝色条代表已经翻译，绿色条代表已经审核。同一条目可以有多条翻译建议（Suggestion）。
翻译者和审核者可以通过投票来表态，最终导出被审核通过的翻译（如果没有审核，则会选择最近的翻译建议）。

Crowdin 不可用时的做法
~~~~~~~~~~~~~~~~~~~~~~

当遇到 Crowdin 平台不可用时，我们可以使用最原始的方式来直接维护 ``.po`` 文件。

假设你发现 `reference/api/megengine.functional.add.html 
<https://megengine.org.cn/doc/stable/zh/reference/api/megengine.functional.add.html>`_
对应的 API Docstring 部分内容翻译有误/没有翻译，标准的处理流程应该如下： 

1. 判断 ``.po`` 文件位置（在这个例子中，属于 API Docstring 英文翻译中文的情况）：
   `locales/zh_CN/LC_MESSAGES/reference/api/megengine.functional.add.po 
   <https://github.com/MegEngine/Documentation/blob/main/locales/zh_CN/LC_MESSAGES/reference/api/megengine.functional.add.po>`_

2. 根据 ``msgstr`` 找到对应位置，根据 ``msgid`` 原文修改 ``msgstr`` 为正确内容；
3. 按照 Git 工作流向 Documentation 库发起 Pull Request.

更多细节请参考 :ref:`contribute-to-docs` 和 :ref:`commit-message` 。

.. note::

   如果不熟悉 Git 操作，你可以通过任何官方渠道与文档维护人员联系并进行反馈，亦可作为 :ref:`共同作者 <doc-co-author>` 。

翻译注意事项
------------

* 语法和排版规范可参考 :ref:`restructuredtext` 和 :ref:`megengine-document-style-guide` 。
* 不要破坏原有的语法格式，正确示范为：

  .. code-block:: po

     #: locales/zh_CN/LC_MESSAGES/example.rst:6
     msgid "A :py:class:`~.megengine.Tensor` object"
     msgstr "一个 :py:class:`~.megengine.Tensor` 对象"

  .. warning::

     Sphinx reStructuredText 语法与周围文本内容之间的空格是 **必需的** ，不然会以文本形式进行渲染 。
     而在 reStructuredText 语法内部，**不该出现空格的地方绝对不能出现空格** ，谨记格式的重要性。
     在对 API 相关内容进行翻译时尤其需要注重这一点，否则将牵一发而动全身。

* 不要自行加入新的 Sphinx reStructuredText 语法，翻译程序将会检测到前后数量不一致。

* 不要丢掉原有标点符号，正确示范为：

  .. code-block:: po

     #: locales/zh_CN/LC_MESSAGES/example.rst:6
     msgid "Method:"
     msgstr "方法："

  当然，多出奇怪的符号也是不允许的。

基本要领
~~~~~~~~

#. 简洁规范。
#. 忠实原文。
#. 用语一致。
#. 易于使用

补充细节说明
~~~~~~~~~~~~

* 中英文之间有且只能有一个空格作为分隔；
* 优先使用全角标点符号（包括逗号、句号、冒号、分号和问号）；
* 遇到特定英文（要求不译）结束，则后跟使用英文标点符号 —— 

  * 绝大部分软件名字都是不翻译的，直接使用英文即可；
  * 项目或组织名称，一般也不进行翻译；
  * 通用的英文缩写，或没有正式中文译文的名词，不需要翻译；

* 不论中英文，统一使用中文括号（）包裹；
* 按回车造成的换行，是 PO 文件里代码换行。
  Crowdin 都能很好地自动换行，因此手动去折行并不是必要的。
  即使是翻译时看到英文是折行的，中文也不一定需要折行；
