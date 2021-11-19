.. highlight:: rst
.. _restructuredtext:

================================
Sphinx reStructuredText 语法入门
================================

本文是一篇符合 MegEngine 文档特色的
《`Sphinx reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext>`_ 》
入门教程，同时也请文档内容的编写人员遵循其中的各项规范。除了这篇供初学者入门的教程以外，
《`reStructuredText User Documentation <http://docutils.sourceforge.net/rst.html>`_ 》
也是一份比较权威的文档，本页面中的所有 **参考** 链接已经指向了权威文档中对应的部分。

.. note::

   * 如果你有 MarkDown 语法经验，学习 reST 语法会更加简单。
   * MegEngine 文档维护人员一定要对 :ref:`超链接 <hyperlinks-rst>` 和 :ref:`交叉引用 <cross-reference-rst>` 的用法烂熟于心。
   * 即使是在 Python 文档字符串中也可以使用 :ref:`sphinx-directives` 和 :ref:`sphinx-ext`.

.. note::

   * 对于简单的文本编辑，可以先在 MarkDown 编辑器中撰写，再通过 Pandoc 转换成 reST 格式；
   * 如果你习惯使用 Visual Studio Code, 可以考虑安装 `reStructuredText Language Support for Visual Studio Code
     <https://marketplace.visualstudio.com/items?itemName=lextudio.restructuredtext>`_ 拓展，所见即所得。

.. warning::
   
   这些指令和自己的可选参数之间不能存在空行，但和被渲染的内容之间必须有空行。

.. _indentation-rst:

缩进（Indentation）
-------------------

.. warning::

   * reST 语法要求使用 3 格缩进（有的是 2 格），错误的缩进将导致无法正确渲染样式。
   * 如果在 Sphinx reST 内使用代码块，其缩进要求不受影响（如 Python）。

对于不同的编辑器，都有着对应的调整缩进宽度的办法（以 Vim 为例子）：

.. code-block:: text

   :set shiftwidth=3
   :set autoindent
   :set smartindent
   :20, 37 >

上面的命令可以快速对 20 至 37 行的内容进行向右缩进。


.. _sections-rst:

章节（Sections）
----------------

通过使用标点符号对节标题加下划线（并可选地对它们进行加上划线）来创建节标题（:duref:`参考 <sections>` ），

MegEngine 采用如下的章节结构语法规范：

.. code-block::

   ========
   一级标题
   ========

   二级标题
   --------

   三级标题
   ~~~~~~~~

.. warning::

   * **标记符必须与文本长度一致，否则会导致 Warning （无法通过 CI）。** 
   * 你可以采用更深的嵌套级别，但在文档中应当避免出现四级甚至更深的标题。

.. panels::

  正确示范
  ^^^^^^^^
  .. code-block::

     ========
     一级标题
     ========

  ---
  错误示范
  ^^^^^^^^
  .. code-block::

     ======================
     一级标题
     ======================

.. _paragraphs-rst:

段落（Paragraphs）
------------------

段落（:duref:`参考 <paragraphs>` ）即一个或多个空白行分隔的文本块，是 reST 文档中最基本的部分。
缩进在 reST 语法中很重要（与 Python 一样），因此同一段落的所有行都必须左对齐到相同的缩进级别。

.. panels::

   没有空行的情况
   ^^^^^^^^^^^^^^
   .. code-block::

      第一部分，
      第二部分。

   第一部分，第二部分。
   ---
   存在空行的情况
   ^^^^^^^^^^^^^^
   .. code-block::

      第一部分，

      第二部分。

   第一部分，
   
   第二部分。

保留换行特性
~~~~~~~~~~~~

行块（:duref:`参考 <line-blocks>` ）是保留换行符的一种方法：

.. code-block::

   | 第一部分，
   | 第二部分。

| 第一部分，
| 第二部分。

.. _inlnie-markup-rst:

内联标记（Inline markup）
-------------------------

包裹使用，标准的 reST 内联标记用法十分简单：


单个星号表示 ``<em>``
  .. code-block:: text

     *HTML 着重元素 (<em>) 标记出需要用户着重阅读的内容*
  
  *HTML 着重元素 (<em>) 标记出需要用户着重阅读的内容*

两个星号表示 ``<strong>``
  .. code-block:: text

     **Strong 元素 (<strong>)表示文本十分重要，一般用粗体显示。**

  **Strong 元素 (<strong>)表示文本十分重要，一般用粗体显示。**

单个反引号表示 ``<cite>``
  .. code-block:: text

     `HTML引用（ Citation）标签 (<cite>) 表示一个作品的引用，且必须包含作品的标题。`

  `HTML引用（ Citation）标签 (<cite>) 表示一个作品的引用，且必须包含作品的标题。`
   
两个反引号表示 ``<pre>``
  .. code-block:: text

     ``HTML <pre> 元素表示预定义格式文本``

  ``HTML <pre> 元素表示预定义格式文本``

.. dropdown:: :fa:`eye,mr-1` 使用注意事项

   .. warning::

      标记符号与被包裹的文本内容之间不能存在空格，与外部文本之间必须存在空格。

   .. panels::

     正确示范
     ^^^^^^^^
     .. code-block:: text

        这些文本 **表示强调** 作用

     这些文本 **表示强调** 作用
     ---
     错误示范
     ^^^^^^^^
     .. code-block:: text

        这些文本 ** 表示强调** 作用
        这些文本 **表示强调 ** 作用
        这些文本**表示强调** 作用

     这些文本 ** 表示强调** 作用
     这些文本 **表示强调 ** 作用
     这些文本**表示强调** 作用

.. _list-rst:

列表（List）
------------

.. warning::

   列表语法是最容易被用错的地方，在文档中也极为常见。

定义列表
~~~~~~~~

定义列表（:duref:`参考 <definition-lists>` ）在 API 文档很常见，使用方法如下：

.. code-block::

   术语 （限定在一行文本）
      术语的定义，必须使用缩进。

      支持使用多个段落。

   下一个术语
      下一个术语对应的定义。

术语 （限定在一行文本）
  术语的定义，必须使用缩进。

  支持使用多个段落。

下一个术语
  下一个术语对应的定义。

无序列表
~~~~~~~~

无序列表（:duref:`参考 <bullet-lists>` ）的用法很自然。
只需要在段落开头放置星号，然后正确地缩进：

.. panels::

   正确的示范（2 格缩进）
   ^^^^^^^^^^^^^^^^^^^^^^
   .. code-block::

      * 这是一个无序列表。
      * 它有两个元素，
        第二个元素占据两行源码，视作同一个段落。

   * 这是一个无序列表。
   * 它有两个元素，
     第二个元素占据两行源码，视作同一个段落。
   ---
   错误的示范（4 格缩进）
   ^^^^^^^^^^^^^^^^^^^^^^
   .. code-block::

      * 这是一个无序列表。
      * 它有两个元素，
          第二个元素被解析成定义列表。

   * 这是一个无序列表。
   * 它有两个元素，
       第二个元素被解析成定义列表。

有序列表
~~~~~~~~

对于有序列表，可以自己编号，也可以使用 # 来自动编号：

.. code-block::

   1. 这是一个有序列表。
   2. 它也有两个元素。

1. 这是一个有序列表。
2. 它也有两个元素。

.. code-block::

   #. 这又是一个有序列表。
   #. 但是它能够自动编号～

#. 这又是一个有序列表。
#. 但是它能够自动编号～


嵌套列表
~~~~~~~~

嵌套列表必须使用空白行和父列表项目隔开：

.. panels::

   正确示范
   ^^^^^^^^
   .. code-block::

      * 这是一个列表。

        * 它嵌套了一个子列表，
        * 并且有自己的子元素。

      * 这里是父列表的后续元素。

   * 这是一个列表。

     * 它嵌套了一个子列表，
     * 并且有自己的子元素。

   * 这里是父列表的后续元素。
   ---
   错误示范
   ^^^^^^^^
   .. code-block::

      * 这并不是嵌套列表，
        * 前面三行被看作是同一个元素，
        * 其中星号被解析成普通的文本。
      * 这是列表的第二个元素。

   * 这并不是嵌套列表，
     * 前面三行被看作是同一个元素，
     * 其中星号被解析成普通的文本。
   * 这是列表的第二个元素。

.. _tables-rst:

表格（Tables）
--------------

网格表
~~~~~~

对于网格表（:duref:`参考 <grid-tables>` ），必须手动“画”出单元格：

.. code-block::

   +------------------------+------------+----------+----------+
   | Header row, column 1   | Header 2   | Header 3 | Header 4 |
   | (header rows optional) |            |          |          |
   +========================+============+==========+==========+
   | body row 1, column 1   | column 2   | column 3 | column 4 |
   +------------------------+------------+----------+----------+
   | body row 2             | ...        | ...      |          |
   +------------------------+------------+----------+----------+

+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | ...        | ...      |          |
+------------------------+------------+----------+----------+

简单表
~~~~~~

简单表（:duref:`参考 <simple-tables>` ）写起来很简单，但有局限性：
它们必须包含多个行，并且第一列单元格不能包含多行。

.. code-block::

   =====  =====  =======
   A      B      A and B
   =====  =====  =======
   False  False  False
   True   False  False
   False  True   False
   True   True   True
   =====  =====  =======

=====  =====  =======
A      B      A and B
=====  =====  =======
False  False  False
True   False  False
False  True   False
True   True   True
=====  =====  =======

CSV 表
~~~~~~

CSV 表格可以根据 CSV（逗号分隔值）数据创建表。

.. code-block::

   .. csv-table:: Frozen Delights!
      :header: "Treat", "Quantity", "Description"
      :widths: 15, 10, 30

      "Albatross", 2.99, "On a stick!"
      "Crunchy Frog", 1.49, "If we took the bones out, 
      it wouldn't becrunchy, now would it?"
      "Gannet Ripple", 1.99, "On a stick!"

.. csv-table:: Frozen Delights!
   :header: "Treat", "Quantity", "Description"
   :widths: 15, 10, 30

   "Albatross", 2.99, "On a stick!"
   "Crunchy Frog", 1.49, "If we took the bones out, 
   it wouldn't becrunchy, now would it?"
   "Gannet Ripple", 1.99, "On a stick!"

List 表
~~~~~~~

List 表可以根据两级无序列表来生成表格：

.. code-block::
   
   .. list-table:: Frozen Delights!
      :widths: 15 10 30
      :header-rows: 1

      * - Treat
        - Quantity
        - Description
      * - Albatross
        - 2.99
        - On a stick!
      * - Crunchy Frog
        - 1.49
        - If we took the bones out, it wouldn't be
          crunchy, now would it?
      * - Gannet Ripple
        - 1.99
        - On a stick!

.. list-table:: Frozen Delights!
   :widths: 15 10 30
   :header-rows: 1

   * - Treat
     - Quantity
     - Description
   * - Albatross
     - 2.99
     - On a stick!
   * - Crunchy Frog
     - 1.49
     - If we took the bones out, it wouldn't be
       crunchy, now would it?
   * - Gannet Ripple
     - 1.99
     - On a stick!

.. _hyperlinks-rst:

超链接（Hyperlinks）
--------------------

使用 ```链接文本 <https://domain.invalid>`_`` 来插入内联网页链接。

你也可以使用目标定义（:duref:`参考 <hyperlink-targets>` ）的形式分离文本和链接：

.. code-block::

   这个段落包含一个 `超链接`_.

   .. _超链接: https://domain.invalid/

这个段落包含一个 `超链接`_.

.. _超链接: https://domain.invalid/

.. warning::

   * 在链接文本和 ``<`` 符号之间必须至少有一个空格。
   * 同 :ref:`inlnie-markup-rst` ，
     标记符和被包裹的文本之间不能有空格，
     而标记符和外部文本之间至少需要有一个空格。


.. _images-rst:

图片（Images）
--------------

reST 支持图像指令，用法如下：

.. code-block::

   .. image:: gnu.png
      :height: 100px (length)
      :width: 200px (length or percentage of the current line width)
      :scale: integer percentage (the "%" symbol is optional)
      :alt: alternate text
      :align: "top", "middle", "bottom", "left", "center", or "right"
      :target: text (URI or reference name)

   当在 Sphinx 中使用时，给定的文件名（在此处为 ``gnu.png`` ）必须相对于源文件。

.. warning::

   * MegEngine 文档中所使用的图片请统一放置在 ``source/_static/images`` 目录内。
     **绝对不允许** 直接将图片放在和文本文件相同的文件夹内，这样虽然方便了写作时进行引用，
     但却给整个文档的维护引入了技术债务，将形成潜在的风险。
   * 一般情况下请优先使用 SVG 格式的矢量图，使用位图请权衡好图片体积和清晰度。
   * 尽可能使用 :ref:`Graphviz <graphviz-ext>` 或 :ref:`Mermaid <mermaid-ext>` 语法绘制示意图。
   * 图片文件名需要有相应的语义信息，不可使用完全随机生成的字符。

.. note::

   如果你想要给图片添加描述性文字，请使用 ``figure`` 代替 ``image``,
   接着使用 ``:caption: text`` 作为传入的参数，其它参数用法一致。

.. _cross-reference-rst:

交叉引用（Cross-reference）
---------------------------

使用 ``:role:`target``` 语法，就会创造一个 ``role`` 类型的指向 ``target`` 的链接。

* 最常见的使用情景是文档内部页面的相互引用（尤其是引用 API 参考内容时）。
* 显示的链接文本会和 ``target`` 一致，
  你也可以使用 ``:role:`title <target>``` 来将链接文本指定为 ``title``

通过 ref 进行引用
~~~~~~~~~~~~~~~~~

为了支持对任意位置的交叉引用，使用了标准的 reST 标签（标签名称在整个文档中必须唯一）。

可以通过两种方式引用标签：

* 在章节标题之前放置一个标签，引用时则可以使用 ``:ref:`label-name``` , 比如：

  .. code-block::

     .. _my-reference-label:

     Section to cross-reference
     --------------------------

     This is the text of the section.

     It refers to the section itself, see :ref:`my-reference-label`.

  这种方法将自动获取章节标题作为链接文本，且对图片和表格也一样有效。

* 如果标签没有放在标题之前，则需要使用 ``:ref:`Link title <label-name>``` 来指定名称。

交叉引用 Python 对象
~~~~~~~~~~~~~~~~~~~~

MegEngine 文档按照 Sphinx `Python Domain <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain>`_ 组织好了 Python API 页面，通常这些信息由 Sphinx 的 ``autodoc`` 插件从 MegEngine Python 接口源码的 docstring 处获得并自动生成。不同的 Python API 的 docstring 之间可以交叉引用，其它类型的文档也可以借此快速跳转到 API 页面。

.. note::

   你可以在 MegEngine 的用户指南文档源码中找到非常多的使用参考。

如果找到匹配的标识符，则会自动生成对应的超链接：

* ``:py:mod:`` 引用一个模块（Module）；可以使用点名。也适用于包（Package）。
* ``:py:func:`` 引用一个 Python 函数；可以使用点名。可不添加括号以增强可读性。
* ``:py:data:`` 引用模块级变量。
* ``:py:const:`` 引用一个 “已定义的” 常量。
* ``:py:class:`` 引用一个类（Class）；可以使用点名。
* ``:py:meth:`` 引用一个对象的方法（Method）；可以使用点名。
* ``:py:attr:`` 引用一个对象的特性（Attribute），也适用于属性（Property）。
* ``:py:exc:`` 引用一个异常（Exception）；可以使用点名。
* ``:py:obj:`` 引用未指定类型的对象。

默认情况下，将在 `当前的模块 
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#directive-py-currentmodule>`_ 中进行搜索。

.. code-block::

   .. currentmodule:: megengine.functional

比如 ``:py:func:`add``` 可以指向当前 ``funtional`` 模块内名为 ``add`` 的一个函数或者 Python Built-in 的函数。
如果使用 ``:py:func:`functional.add``` 则可以明确指向到 ``functional`` 模块中的 ``add`` 函数。

如果使用前缀 ``~`` , 链接文本将会只显示 ``target`` 的最后一部分。
例如 ``:py:func:`~megengine.functional.add``` 将会指向 ``megengine.functional.add``
但显示为 :py:func:`~megengine.functional.add` .

如果使用 ``.`` 点名，在没有找到完全匹配的内容时，会将点名作为后缀，
并开始搜索和匹配带有该后缀的所有对象的名称（即使匹配到的结果不在当前模块）。
例如在已知当前模块为 ``data`` 时，使用 ``:py:func:`.functional.add``` 
会找到 :py:func:`.functional.add` . 我们也可以结合使用 ``~`` 和 ``.`` ，
如 ``:py:func:`~.functional.add``` 将只显示 :py:func:`~.functional.add` .

.. warning::

   MegEngine 文档列举出的 Python API 通常是最短调用路径。
   比如 ``add`` 的实际路径是 ``megengine.functional.elemwise.add`` ，
   但在文档中能够搜索到的路径只有 ``megengine.functional.add`` .
   因此在引用时应当使用 ``:py:func:`~.functional.add``` 而不是 ``:py:func:`~.functional.elemwise.add``` 
   （后者会因为匹配失败而无法生成超链接）。

.. warning::

   对于 ``Conv2d`` 这种存在多个前缀的 API, 请补全部分前缀以防止冲突，比如 ``:py:class:`.module.Conv2d```.

.. _footnotes-rst:

脚注（Footnotes）
-----------------

脚注（:duref:`参考 <footnotes>` ）使用 ``[#name]_`` 来标记脚注的位置，并在 ``Footnotes`` 专栏（rubic）后显示，例如：

.. code-block::

   Lorem ipsum [1]_ dolor sit amet ... [2]_

   .. rubric:: Footnotes

   .. [1] Text of the first footnote.
   .. [2] Text of the second footnote.

Lorem ipsum [1]_ dolor sit amet ... [2]_

.. rubric:: Footnotes

.. [1] Text of the first footnote.
.. [2] Text of the second footnote.

你可以显式使用 ``[1]_`` 来编号，否则使用 ``[#]_`` 进行自动编号。

.. _citation-rst:

引用（Citation）
----------------

引用和脚注类似，但不需要进行编号，且全局可用：

.. code-block::

   Lorem ipsum [Ref]_ dolor sit amet.

   .. [Ref] Book or article reference, URL or whatever.

Lorem ipsum [Ref]_ dolor sit amet.

.. [Ref] Book or article reference, URL or whatever.

.. _math-rst:

数学公式（Math）
----------------

只需要使用类似的语法：

.. code-block::

   Since Pythagoras, we know that :math:`a^2 + b^2 = c^2`.

.. _sphinx-directives:

Sphinx 拓展指令
---------------

.. warning::

   以下语法非原生 ReStructuredText 语法，需要通过 Sphinx 进行支持。

``.. toctree::`` 
  Table of contents tree. 用于组织文档结构。

``.. note::`` 
  用于添加提示性信息，用户忽视这些信息可能出错。
  
``.. warning::``
  用于添加警告性信息，用户忽视这些信息一定出错。

``.. versionadded:: version``
  描述 API 添加版本，如果用于单个模块, 则必须放在显式文本内容顶部。

``.. versionchanged:: version``
  描述 API 变更版本，指出以何种方式（新参数）进行了更改以及可能的副作用。

``.. deprecated:: version``
  描述 API 弃用版本，简要地告知替代使用方式。

``.. seealso::``
  包括对模块文档或外部文档的引用列表，内容应该是一个 reST 定义列表，比如：
  
  .. code-block::

     .. seealso::

     Module :py:mod:`zipfile`
        Documentation of the :py:mod:`zipfile` standard module.

     `GNU tar manual, Basic Tar Format <http://link>`_
        Documentation for tar archive files, including GNU tar extensions.

  也可以使用简略写法，如下所示：

  .. code-block::
     
     .. seealso:: modules :py:mod:`zipfile`, :py:mod:`tarfile`

``.. rubric:: title``
  用于创建一个不会产生导航锚点的标题。

``.. centered::``
  创建居中文本

``.. math::``
  LaTeX 标记的数学公式，相较于 ``:math:`` 语法提供了更干净的阅读空间。

  .. code-block::

     .. math::

      (a + b)^2 = a^2 + 2ab + b^2

      (a - b)^2 = a^2 - 2ab + b^2

  .. math::

   (a + b)^2 = a^2 + 2ab + b^2

   (a - b)^2 = a^2 - 2ab + b^2

  .. warning::

     用于 Python 文档字符串中时，必须将所有反斜杠加倍，或者使用 Python 原始字符串 ``r"raw"``.

``.. highlight:: language``
  使用指定语言（Pygments 支持）的语法高亮，直到再次被定义。

``.. code-block:: [language]``
  展示代码块，如果未设置 ``language``, highlight_language 将被使用。
  
.. note::

   想要了解完整的指令和配置项，请访问 `Directives
   <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_ 页面。

.. _sphinx-ext:

Sphinx 插件语法
---------------

.. note::

   下面的语法通过 Sphinx Extensions 支持，同样可以用于 Python 文档字符串。

.. _bibtex-ext:

Bibtex 语法支持
~~~~~~~~~~~~~~~

.. _graphviz-ext:

文档已经通过 `sphinxcontrib-bibtex 
<https://sphinxcontrib-bibtex.readthedocs.io/en/latest/index.html>`_ 插件支持
`Bibtex <http://www.bibtex.org/>`_ 语法。

Graphviz 语法支持
~~~~~~~~~~~~~~~~~

文档已经通过 `sphinx.ext.graphviz 
<https://www.sphinx-doc.org/en/master/usage/extensions/graphviz.html>`_ 插件支持
`Graphviz <https://graphviz.org/>`_ 语法，样例如下：

.. code-block:: 

   .. graphviz::

      digraph foo {
         "bar" -> "baz";
      }


.. graphviz::

   digraph foo {
      "bar" -> "baz";
   }

.. _mermaid-ext:

Mermaid 语法支持
~~~~~~~~~~~~~~~~

文档已经通过 `sphinxcontrib-mermaid 
<https://sphinxcontrib-mermaid-demo.readthedocs.io/en/latest/>`_ 插件支持
`Mermaid <https://mermaid-js.github.io/mermaid/>`_ 语法，样例如下：

.. code-block::
   
   .. mermaid::

      sequenceDiagram
         participant Alice
         participant Bob
         Alice->John: Hello John, how are you?
         loop Healthcheck
             John->John: Fight against hypochondria
         end
         Note right of John: Rational thoughts <br/>prevail...
         John-->Alice: Great!
         John->Bob: How about you?
         Bob-->John: Jolly good!

.. mermaid::

   sequenceDiagram
      participant Alice
      participant Bob
      Alice->John: Hello John, how are you?
      loop Healthcheck
          John->John: Fight against hypochondria
      end
      Note right of John: Rational thoughts <br/>prevail...
      John-->Alice: Great!
      John->Bob: How about you?
      Bob-->John: Jolly good!

.. _toggle-ext:

Toggle 语法支持
~~~~~~~~~~~~~~~

文档已经通过 `sphinx-togglebutton 
<https://sphinx-togglebutton.readthedocs.io/en/latest/>`_ 插件支持常见 Toggle 功能，样例如下：

.. code-block::

   .. admonition:: Here's my title
      :class: dropdown, warning

      My note

.. admonition:: Here's my title
   :class: dropdown, warning
   
   My note

以上展示的为基础用法，更多用法请参考文档。

.. _pannels-ext:

Pannels 语法支持
~~~~~~~~~~~~~~~~

文档已经通过 `sphinx-panels 
<https://sphinx-panels.readthedocs.io/en/latest/>`_ 插件支持常见 Pannels 功能，样例如下：

.. code-block::

   .. panels::
      :container: container-lg pb-3
      :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

      panel1
      ---
      panel2
      ---
      panel3
      ---
      :column: col-lg-12 p-2
      panel4

.. panels::
   :container: container-lg pb-3
   :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

   panel1
   ---
   panel2
   ---
   panel3
   ---
   :column: col-lg-12 p-2
   panel4

以上展示的为 Grid Layout 用法，Card Layout, Image Caps 等用法请参考文档。

.. note::

   该插件也实现了 Toggle, Tabs 语法功能。

.. _tabs-ext:

Tabs 语法支持
~~~~~~~~~~~~~

文档已经通过 `sphinx-tabs 
<https://sphinx-tabs.readthedocs.io/en/latest/>`_ 插件支持常见 Tabs 功能，样例如下：

.. code-block::

   .. tabs::

      .. tab:: Apples

         Apples are green, or sometimes red.

      .. tab:: Pears

         Pears are green.

      .. tab:: Oranges

         Oranges are orange.

.. tabs::

   .. tab:: Apples

      Apples are green, or sometimes red.

   .. tab:: Pears

      Pears are green.

   .. tab:: Oranges

      Oranges are orange.

以上展示的为 Basic 用法，Nested / Group / Code Tabs 用法请参考文档。

GitHub URL 缩写
~~~~~~~~~~~~~~~

为了方面写文档时引用 GitHub 上的源代码，支持如下语法：

.. code-block:: 
   
   * :src:`imperative/python/megengine/`
   * :docs:`source/conf.py`
   * :issue:`142`
   * :pr:`148`

* :src:`imperative/python/megengine/`
* :docs:`source/conf.py`
* :issue:`142`
* :pull:`148`

该功能通过 `sphinx.ext.extlinks 
<https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html>`_ 插件支持。

