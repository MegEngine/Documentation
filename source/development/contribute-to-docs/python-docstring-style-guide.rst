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

   在 MegEngine 源码中鼓励使用 Google 风格的文档字符串。 (必须带上类型提示)

.. warning::

   * 由于历史原因，MegEngine 曾选择了使用 reStructuredText 文档字符串风格； 
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

除此以外，开发者在为 MegEngine 的 Python API 编写 Docstring 时，还需注意以下情况：

#. **Tensor API 文档字符串优先参考《数组 API 标准》。** 在《 :ref:`mep-0003` 》中，明确了 MegEngine Tensor API 在设计时将尽量参考
   《 `数组 API 标准 <https://data-apis.org/array-api/latest/>`_ 》中所定义的规格和约定，文档字符串也应当遵循这一原则。
   当某个 Tensor API 已经存在于《标准》之中时，文档字符串编辑人员应当仔细确认《标准》中所陈述的行为在 MegEngine 中表现是否一致。
   对于完全一致的行为，应当使用一致的、《标准》中已经提供的文档字符串进行描述；对于不一致的行为，应当以提示或警告的形式进行说明。

#. **可适当重写以覆盖 API 源码中提供的类型提示。** 默认情况下，API 文档中的类型提示将根据源码 TypeHint 内容生成。
   一些仅内部使用的类型如 ``SymbolVar``, ``CompNote``... 所涉及的概念会让用户感到迷惑，此时应当在文档字符串中适当重写。
   做法是在参数后面空一格，然后用半角圆括号括起重写后的类型提示内容：

   .. code-block:: python
      :emphasize-lines: 5

      def func(inp: Union[Tensor, SymbolVar]) -> Union[Tensor, SymbolVar]:  # <- TypeHint
          r""""Example function with PEP 484 type annotations.

          Args:
              inp (Tensor): The input tensor.

          Returns:
              The return tensor.
          """
          pass

   理想状态下，源码中的每个 API 的参数都应该带上类型提示，这样做对编辑器、集成开发环境更为友好。
   覆写 TypeHint 内容会引入额外的维护成本，因此不建议所有的 TypeHint 都进行人为覆写。

#. **示范代码必须使用标准 doctest 风格而非 code-output 风格。** 对比如下：

   .. panels::
      :container: +full-width
      :card:

      错误写法
      ^^^^^^^^
      .. code-block::

         .. testcode::

            import megengine.functions as F

            a = F.arange(5)
            print(a.numpy)

         .. testoutput::

            [0. 1. 2. 3. 4.]
   
      ---
      正确写法
      ^^^^^^^^
      .. code-block::

         >>> megengine.functional.arange(5)
         Tensor([0. 1. 2. 3. 4.], device=xpux:0)

         >>> megengine.functional.arange(1, 4)
         Tensor([1. 2. 3.], device=xpux:0)

      * 一些时候可用注释代替上下文；
      * 可以有多例，以展示不同的用法。


#. **API 文档首行简述应确保做到 “清晰、准确、概括” 这三点要求。** 错误例子如下：

   .. code-block:: python

      def all_reduce_max(...):
          r"""Create all_reduce_max operator for collective communication."""

   整个文档字符串的内容只有上述这句话，对于一个不了解分布式概念的用户来说， 仅提供这些信息的帮助极其有限。
   用户完全不知道这样的 API 能够用在什么地方，也有可能对 “聚合式通信（collective communication）” 的概念一无所知。
   我们希望文档中所提供的概念应该是自包含的（Self-contained），解释性的文本（或对应的链接）是不可或缺的，尽可能避免让用户去搜索其它材料。

   对于一些比较复杂的、或需要结合情景使用的 API, 仅靠示范代码不足以帮助用户理解使用情景，
   此时可以提供到专门介绍用法的文档页面的链接。常见的做法是使用 ``seealso`` 进行拓展：

   .. code-block:: restructuredtext

      .. seealso::

         See :ref:`collective-communication-intro` for more details.


#. **合理补充提示（Note）和警告（Warning）信息，善用页面交叉引用，根据用户反馈不断迭代。** 


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


