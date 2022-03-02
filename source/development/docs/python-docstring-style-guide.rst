.. _python-docstring-style-guide:

=========================
Python 文档字符串风格指南
=========================

如果你对 Python 文档字符串（Docstring）的概念不是很清楚，可以参考以下材料：

* `PEP 257 <https://www.python.org/dev/peps/pep-0257>`_ - 文档字符串约定（Docstring Conventions）
* `PEP 287 <https://www.python.org/dev/peps/pep-0287>`_ - reStructuredText 风格文档字符串格式
* `PEP 484 <https://www.python.org/dev/peps/pep-0484>`_ - 类型提示（Type Hints）
* `Google Python Style guides <https://google.github.io/styleguide/pyguide.html#381-docstrings>`_ - Google 风格文档字符串格式

.. note::

   在 MegEngine 源码中要求统一使用 Google 风格的文档字符串。

.. warning::

   * 由于历史原因，MegEngine 曾选择了使用 reStructuredText 文档字符串风格描述参数与返回值；
   * 在 >=1.6 版本的 MegEngine, 所有文档字符串将统一迁移成 Google 风格；
   * 如果你发现了 MegEngine 仍然存在 ReST 风格的参数/返回值写法，欢迎帮我们改正过来。

.. _docstring-template:

Docstring Google Style 模版
---------------------------

我们建议所有刚开始尝试 Docstring 编写的开发者看一看下面这个模版文件
（ `源文件地址 <https://github.com/sphinx-contrib/napoleon/blob/master/docs/source/example_google.py>`_ ）：

.. dropdown:: Example Google style docstrings

   .. code-block:: python

      r"""Example Google style docstrings.

      This module demonstrates documentation as specified by the `Google Python
      Style Guide`_. Docstrings may extend over multiple lines. Sections are created
      with a section header and a colon followed by a block of indented text.

      Example:
          Examples can be given using either the ``Example`` or ``Examples``
          sections. Sections support any reStructuredText formatting, including
          literal blocks::

              $ python example_google.py

      Section breaks are created by resuming unindented text. Section breaks
      are also implicitly created anytime a new section starts.

      Attributes:
          module_level_variable1 (int): Module level variables may be documented in
              either the ``Attributes`` section of the module docstring, or in an
              inline docstring immediately following the variable.

              Either form is acceptable, but the two should not be mixed. Choose
              one convention to document module level variables and be consistent
              with it.

      Todo:
          * For module TODOs
          * You have to also use ``sphinx.ext.todo`` extension

      .. _Google Python Style Guide:
         http://google.github.io/styleguide/pyguide.html

      """

      module_level_variable1 = 12345

      module_level_variable2 = 98765
      r"""int: Module level variable documented inline.

      The docstring may span multiple lines. The type may optionally be specified
      on the first line, separated by a colon.
      """


      def function_with_types_in_docstring(param1, param2):
          r"""Example function with types documented in the docstring.

          `PEP 484`_ type annotations are supported. If attribute, parameter, and
          return types are annotated according to `PEP 484`_, they do not need to be
          included in the docstring:

          Args:
              param1 (int): The first parameter.
              param2 (str): The second parameter.

          Returns:
              bool: The return value. True for success, False otherwise.

          .. _PEP 484:
              https://www.python.org/dev/peps/pep-0484/

          """


      def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
          r"""Example function with PEP 484 type annotations.

          Args:
              param1: The first parameter.
              param2: The second parameter.

          Returns:
              The return value. True for success, False otherwise.

          """


      def module_level_function(param1, param2=None, *args, **kwargs):
          r"""This is an example of a module level function.

          Function parameters should be documented in the ``Args`` section. The name
          of each parameter is required. The type and description of each parameter
          is optional, but should be included if not obvious.

          If \*args or \*\*kwargs are accepted,
          they should be listed as ``*args`` and ``**kwargs``.

          The format for a parameter is::

              name (type): description
                  The description may span multiple lines. Following
                  lines should be indented. The "(type)" is optional.

                  Multiple paragraphs are supported in parameter
                  descriptions.

          Args:
              param1 (int): The first parameter.
              param2 (:obj:`str`, optional): The second parameter. Defaults to None.
                  Second line of description should be indented.
              *args: Variable length argument list.
              **kwargs: Arbitrary keyword arguments.

          Returns:
              bool: True if successful, False otherwise.

              The return type is optional and may be specified at the beginning of
              the ``Returns`` section followed by a colon.

              The ``Returns`` section may span multiple lines and paragraphs.
              Following lines should be indented to match the first line.

              The ``Returns`` section supports any reStructuredText formatting,
              including literal blocks::

                  {
                      'param1': param1,
                      'param2': param2
                  }

          Raises:
              AttributeError: The ``Raises`` section is a list of all exceptions
                  that are relevant to the interface.
              ValueError: If `param2` is equal to `param1`.

          """
          if param1 == param2:
              raise ValueError('param1 may not be equal to param2')
          return True


      def example_generator(n):
          r"""Generators have a ``Yields`` section instead of a ``Returns`` section.

          Args:
              n (int): The upper limit of the range to generate, from 0 to `n` - 1.

          Yields:
              int: The next number in the range of 0 to `n` - 1.

          Examples:
              Examples should be written in doctest format, and should illustrate how
              to use the function.

              >>> print([i for i in example_generator(4)])
              [0, 1, 2, 3]

          """
          for i in range(n):
              yield i


      class ExampleError(Exception):
          r"""Exceptions are documented in the same way as classes.

          The __init__ method may be documented in either the class level
          docstring, or as a docstring on the __init__ method itself.

          Either form is acceptable, but the two should not be mixed. Choose one
          convention to document the __init__ method and be consistent with it.

          Note:
              Do not include the `self` parameter in the ``Args`` section.

          Args:
              msg (str): Human readable string describing the exception.
              code (:obj:`int`, optional): Error code.

          Attributes:
              msg (str): Human readable string describing the exception.
              code (int): Exception error code.

          """

          def __init__(self, msg, code):
              self.msg = msg
              self.code = code


      class ExampleClass(object):
          r"""The summary line for a class docstring should fit on one line.

          If the class has public attributes, they may be documented here
          in an ``Attributes`` section and follow the same formatting as a
          function's ``Args`` section. Alternatively, attributes may be documented
          inline with the attribute's declaration (see __init__ method below).

          Properties created with the ``@property`` decorator should be documented
          in the property's getter method.

          Attributes:
              attr1 (str): Description of `attr1`.
              attr2 (:obj:`int`, optional): Description of `attr2`.

          """

          def __init__(self, param1, param2, param3):
              r"""Example of docstring on the __init__ method.

              The __init__ method may be documented in either the class level
              docstring, or as a docstring on the __init__ method itself.

              Either form is acceptable, but the two should not be mixed. Choose one
              convention to document the __init__ method and be consistent with it.

              Note:
                  Do not include the `self` parameter in the ``Args`` section.

              Args:
                  param1 (str): Description of `param1`.
                  param2 (:obj:`int`, optional): Description of `param2`. Multiple
                      lines are supported.
                  param3 (:obj:`list` of :obj:`str`): Description of `param3`.

              """
              self.attr1 = param1
              self.attr2 = param2
              self.attr3 = param3  #: Doc comment *inline* with attribute

              #: list of str: Doc comment *before* attribute, with type specified
              self.attr4 = ['attr4']

              self.attr5 = None
              r"""str: Docstring *after* attribute, with type specified."""

          @property
          def readonly_property(self):
              r"""str: Properties should be documented in their getter method."""
              return 'readonly_property'

          @property
          def readwrite_property(self):
              r""":obj:`list` of :obj:`str`: Properties with both a getter and setter
              should only be documented in their getter method.

              If the setter method contains notable behavior, it should be
              mentioned here.
              """
              return ['readwrite_property']

          @readwrite_property.setter
          def readwrite_property(self, value):
              value

          def example_method(self, param1, param2):
              r"""Class methods are similar to regular functions.

              Note:
                  Do not include the `self` parameter in the ``Args`` section.

              Args:
                  param1: The first parameter.
                  param2: The second parameter.

              Returns:
                  True if successful, False otherwise.

              """
              return True

          def __special__(self):
              r"""By default special members with docstrings are not included.

              Special members are any methods or attributes that start with and
              end with a double underscore. Any special member with a docstring
              will be included in the output, if
              ``napoleon_include_special_with_doc`` is set to True.

              This behavior can be enabled by changing the following setting in
              Sphinx's conf.py::

                  napoleon_include_special_with_doc = True

              """
              pass

          def __special_without_docstring__(self):
              pass

          def _private(self):
              r"""By default private members are not included.

              Private members are any methods or attributes that start with an
              underscore and are *not* special. By default they are not included
              in the output.

              This behavior can be changed such that private members *are* included
              by changing the following setting in Sphinx's conf.py::

                  napoleon_include_private_with_doc = True

              """
              pass

          def _private_without_docstring(self):
              pass

.. note::

   * 阅读 :ref:`document-reference` 会对了解如何组织内容有所帮助。
   * 上面给出的样例模版更多地是作为形式上的参考，适合作为比对。

Docstring 撰写指南
------------------

在 《Google Python Style Guide》的第 `3.8 <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ 小节，
已经提供了相当丰富的建议，如：

* 函数（方法、或生成器）必须提供文档字符串，除非它：对外不可见、很短、用途明显；
* 文档字符串应该提供足够的信息来体现函数的调用方式，使用户无需阅读其源码即可使用；
* 文档字符串应描述函数的调用语法及其语义，但通常不描述其实现细节；
* 文档字符串应该是描述性风格而不是命令式风格... 等等。

格式化检查与排版预览
~~~~~~~~~~~~~~~~~~~~

在提交修改之前，可以使用 MegEngine 自带的脚本进行 Python 代码格式化检查：

.. code-block:: shell
   
   ./imperative/python/scripts/format.sh

语法 & 格式正确是文档基本底线要求，不能在构建过程中引入 Warning 信息。

.. seealso::

   * :ref:`restructuredtext` —— 掌握 ReST 语法的基础样式排版要求
   * :ref:`how-to-build-the-doc-locally` —— 验证你的排版是否符合预期视觉效果

除此以外，开发者在为 MegEngine 的 Python API 编写 Docstring 时，其内容必须满足下列要求：

与社区标准统一，遵循使用许可
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   在撰写 Docstring 的正式内容之前，通常我们有以下参考源头：

   1. 形成广泛共识的社区标准文件，亦或是国际及世界各国的规格/标准组织材料；
   2. 当前主流框架、库（如 NumPy, PyTorch 等）中提供的类似接口的注释；
   3. 来自已经出版的文献材料中的定义与描述。

   优先使用第 1 类材料作为最佳实践，其内容可被直接用于文档字符串中，或适当进行修改；
   对于第 2 类材料，其内容仅适合作为参考， **严禁直接复制粘贴受许可协议保护的代码注释；** 
   对于第 3 类材料，应当充分理解其概念与在 MegEngine 中的接口设计背景，再添加相应注释与引用。

在《 :ref:`mep-0003` 》中，明确了 MegEngine Tensor API 块在设计与维护时，将尽量参考
《 `数组 API 标准 <https://data-apis.org/array-api/latest/>`_ 》中所定义的规格和约定提供接口，文档字符串也应当遵循这一原则。
当发现相应 API 已经存在于《标准》中时，文档字符串的编辑人员应当仔细确认《标准》中所陈述的行为在 MegEngine 中表现是否一致 ——

* 对于完全一致的行为，应当使用《标准》中已经提供的文档字符串内容进行描述；
* 对于不一致的行为，应当收集相关信息并同审核人员讨论，最终需以提示或警告的形式进行说明。

其中与《标准》不一致的行为包括但不限于以下几种：

* 参数选项、命名不一致，仅位置（Position-obly）和仅关键字（Keyword-only）参数的划分不一致；
* 对数据类型的支持情况不一致（通常需要指向 :ref:`tensor-dtype` 页面）；
* 同一使用环境下，使用相同接口、参数，最终效果与预期标准不一致。

对于未在《标准》中的定义 API, 其参数命名与描述风格也应当尽可能与 MegEngine 的整体风格保持一致。

.. _docstring-typehint:

酌情覆盖 PEP 484 类型提示
~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   **这是少数情况。**
   默认情况下，API 文档中的类型提示将按照 `PEP 484 <https://www.python.org/dev/peps/pep-0484>`_ 
   格式从 API 函数或方法的签名中生成，方便一些编辑器做代码跳转和提示，但通常不具备语义描述。
   这种自动生成的类型提示在 Web 文档中样式不够友好，并且对于一些仅内部接口使用到的类型如 ``SymbolVar`` ...
   这些类型是我们不希望用户在使用相关接口时去注意到的，因此要求在文档字符串中对类型提示进行覆盖。

具体做法是：在参数后面空一格，然后用半角圆括号括起重写后的类型提示内容：

.. code-block:: python
   :emphasize-lines: 5

   def func(inp: Union[Tensor, SymbolVar]) -> Union[Tensor, SymbolVar]:
       r""""Example function with PEP 484 type annotations.

       Args:
           inp (Tensor): The input tensor.

       Returns:
           The return tensor.
       """
       pass

.. warning::

   上例中的类型提示表明了 —— 我们期望用户传入的是 ``Tensor`` 类型，但不是限制用户仅能够传入 ``Tensor`` 类型。
   如果用户传入了其它类型数据比如 ``list of ints`` 等，其能够被转换成预期的 ``Tensor`` 类型并被正常使用。
   只是这些类型也不该出现在类型提示中，因为使用它们作为参数是非预期的行为。

   另一个典型是 :attr:`.Tensor.shape`, 它本身在一些情景下类型可以是一个 :class:`~.Tensor`,
   但在大部分接口中其作为参数时的类型提示应当是 ``int or sequence of ints``. 总原则是：一切以推荐使用情景为前提。

下面是一些常被用于覆盖原有提示信息的参数以及对应类型举例：

* ``input/output (Tensor, optional)`` - 将会指向 :class:`~.Tensor`, 视实际情况添加 ``optional``.
* ``inppus/outputs (sequence of Tensors)`` - 表明输入应当是由 ``Tensor`` 组成的序列。
* ``shape (int or sequence of ints)`` - 表明可以是单个 ``int``, 也可以是 ``int`` 组成的序列
* ``xxx (Number)`` - 通常覆盖 ``Union[int, float, Tensor]``
* ``yyy (Number, optional)`` - 通常覆盖 ``Union[int, float, Tensor, None]``
* ``dtype (:attr:`.Tensor.dtype`, optional)`` - 将会指向 :attr:`.Tensor.dtype`
* ``device (:attr:`.Tensor.device`, optional)`` - 将会指向 :attr:`.Tensor.device`
* 对于类型为字符串 :py:class:`str` 的参数，通常可以在类型提示中用更具体的语义对类型进行描述。

对于不确定是否要覆盖 PEP 484 类型提示的情景，需要找代码审核人员进行讨论。

.. _docstring-example:

提供简明而全面的示范代码
~~~~~~~~~~~~~~~~~~~~~~~~

所有的 API Example 必须使用标准 `doctest <https://docs.python.org/3/library/doctest.html>`_ 风格：

>>> F.arange(5)
Tensor([0. 1. 2. 3. 4.], device=xpux:0)

>>> F.arange(1, 4)
Tensor([1. 2. 3.], device=xpux:0)

简明是第一准则，多余的上下文准备操作可以用注释进行说明，一些情景下允许使用伪代码展示用法。
全面指的是：一些参数的变化可能会导致用法改变，此时要提供多例 API 示范代码帮助用户理解。
如果有必要的话，在单个的 API 文档页面中可以引用相应的用户指南，甚至是教程页面。

.. note::

   * 一些 ``import`` 语句可以省略，参考 :src:`imperative/python/megengine/conftest.py` 中的规则；
   * 能够使用 MegEngine API 生成的数据，尽可能避免使用 NumPy API 来生成（除 ``random`` 以外）；
   * 对于位置参数，统一看作是仅位置参数（将来可能会变成强制要求），使用时不要带上参数名；
   * 示范代码的输入输出要求必须一致，否则将无法通过 CI 中的相关测试。常见的错误如下：

     >>> F.eye(3)
     Tensor([[1. 0. 0.]
             [0. 1. 0.]
             [0. 0. 1.]], device=xpux:0)

     这里为了使得元素对齐，人为加入了多余的空格，实际上测试期望得到的输出应当是：

     >>>  F.eye(3)
     Tensor([[1. 0. 0.]
      [0. 1. 0.]
      [0. 0. 1.]], device=xpux:0)

.. warning::

   MegEngine 源码中可能还有一些 `code-output <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ 
   风格的示范代码，大都如下形式：
  
      
   .. code-block::

      .. testcode::

         import megengine.functions as F

         a = F.arange(5)
         print(a.numpy)

      .. testoutput::

         [0. 1. 2. 3. 4.]
    
   这种风格是历史写法，在源码中占据了太多空间，不够简洁，发现之后应该修改成标准 doctest 形式。

.. docstring-special-case:

对特殊情况进行说明
~~~~~~~~~~~~~~~~~~

一些 API 在特殊情况下行为可能发生变化，或者一些情况下的效果需要做更加进一步说明。例如：

.. note::

   * 在参数 A 和 B 的关系满足 xxx 条件时，执行逻辑会变成 xxxxxx 情况；
   * 想要使用本接口，还需要设置 XXX 环境变量。

.. warning::

   使用本接口可能会导致速度变慢！

Docstring 对文档的意义
----------------------

.. admonition:: API 参考页面自动生成
   :class: note

   我们借助 Sphinx 来构建整个 MegEngine 文档（参考 :ref:`how-to-build-the-doc-locally` ），
   其中每个 Python API 的单个文档页面（如 :func:`~.functional.add` ）都是提取相应的文档字符串内容自动生成的。

   前面提到了，在 MegEngine 源码中鼓励使用 Google 风格的文档字符串。
   由于 Sphinx 在根据文档字符串生成 API 页面时，默认只支持 reStructuredText 语法。
   因此我们用到了 `sphinx.ext.napoleon 
   <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ 插件，
   它能够在生成文档前临时将所有的 Google Style 语法解析成 reStructureText 语法。
   这也意味着我们依旧可以使用 :ref:`restructuredtext` 中提到的各种语法来编辑文档字符串内容，
   包括 API 之间的交叉引用、超链接、插入图片，甚至加入一些高级的 HTML 视觉样式。

   但是，我们也要考虑到习惯直接阅读源码（以及使用 ``help()`` / ``print(*.__doc__)`` 语法）的用户，
   使用过多的衍生语法和交叉引用将打破纯文本样式的约定，降低 Python 源码的阅读体验，需克制使用。

.. admonition:: 别忘记提供对应的翻译文本
   :class: warning

   MegEngine 文档的特点之一是提供了中文 API 翻译，而 Docstring 作为源代码的一部分，当然是用英文撰写的。
   因此在 MegEngine 源代码中修改 Docstring 后我们还需要在 Documentation 文档中更新对应 ``.po`` 文件，
   Sphinx 在生成文档时会先检索匹配到的文本，接着自动地将原文替换成对应的译文，这和 WordPress 国际化原理类似，
   相关细节和翻译流程请参考 :ref:`translation` 。


