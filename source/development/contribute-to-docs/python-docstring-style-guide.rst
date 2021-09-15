.. _python-docstring-style-guide:

=========================
Python 文档字符串风格指南
=========================

如果你对 Python 文档字符串（Docstring）的概念不是很清楚，可以参考以下材料：

* `PEP 257 <https://www.python.org/dev/peps/pep-0257>`_ - Docstring Conventions
* `PEP 287 <https://www.python.org/dev/peps/pep-0287>`_ - reStructuredText Docstring Format
* `PEP 484 <https://www.python.org/dev/peps/pep-0484>`_ - Type Hints
* `Google Python Style guides <https://google.github.io/styleguide/pyguide.html#381-docstrings>`_ - Docstrings

.. note::

   在 MegEngine 源码中鼓励使用 Google 风格的 Docstring. (必须带上 Type Hints)

.. warning::

   * 由于历史原因，MegEngine 曾选择了使用 reStructuredText Docstring 代码风格。 
   * 在 >=1.6 版本的 MegEngine, 所有 Docstring 将统一迁移成 Google Style.

.. _docstring-template:

Docstring Google Style 模版
---------------------------

我们建议所有刚开始尝试 Docstring 编写的开发者看一看下面这个模版文件
（ `源文件地址 <https://github.com/sphinx-contrib/napoleon/blob/master/docs/source/example_google.rst>`_ ）：

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

Docstring 编辑自查清单
----------------------

* 一份好的 Docstring 应当具有 Args, Returns, Examples 等章节；

* 对于具有返回值的 APIs, 请统一按照 
  `PEP 484 <https://www.python.org/dev/peps/pep-0484>`_ 添加类型提示 Type Hints;

* Google Style 语法请统一使用与 Python 一致的 4 格缩进（而非 2 格），
  另一种常见的错误写法如下：

  .. code-block::

     Example:
     .. code-block::

        example_func()

  对于一些指令，我们必须将其内容域向右缩进 4 格以便识别：

  .. code-block::

     Example:
         .. code-block::

            example_func()

* 通常而言，我们应该在 API Examples 中向用户展示如何正确地使用 APIs,
  而不是用一些单元测试用例作为使用示范，因为一些时候这只能验证接口的有效性，
  无法帮助查阅 API 文档的用户理解如何去使用它们；因此请勿使用 ``testcode``.
  如果你必须在解释清楚一些相关概念或实现常见任务后才能向用户说明 API 用法，
  通常表明你需要提供一份教程或者是用户指南来进行额外的说明，并在 API 参考中引用。
  
* 请花时间琢磨一下：文档字符串中的首行是否起到了 **清晰、准确、概括** 的效果，错误例子如下：

  .. code-block:: python

     def all_reduce_max(...):
         r"""Create all_reduce_max operator for collective communication."""

  整个注释内容只有上述这句话，对于一个不了解分布式概念的用户来说，
  仅提供这些信息的注释是毫无作用的，甚至让人感到更加的迷惑。
  用户完全不知道这样的 API 能够用在什么地方，也不知道什么是 collective communication.
  我们应当认为该 API 文档字符串并没有起到应该有的作用，形同虚设。

  参考解决办法如下：

  * 如果有且仅有一处相关的 API 实现（常见于某个算法实现调用接口），
    则需要在文档字符串中对相关的概念和算法进行简单解释，
    如果有必要的话，需要给出相关参考材料的原始出处的引用；
  * 如果存在着一类属于相同概念但细节不同的 APIs, 则需要提供用户指南集中介绍背景，
    可以参考 NumPy 中对概念
    `Discrete Fourier Transform <https://numpy.org/doc/stable/reference/routines.fft.html>`_ 
    的解释，以及参考 API 文档如 :py:func:`numpy.fft.fft`. 

  对于上面这个例子，我们应该有一个页面对分布式通信和计算的概念进行介绍，并提供相关例程。


Docstring 对文档的意义
----------------------

一些开发者的 Python 开发环境中，可以根据 Docstring 进行更加智能的内容提示。但这还不是全部，

在 MegEngine 文档中，Docstring 肩负着重要的使命——

API 参考页面自动生成
~~~~~~~~~~~~~~~~~~~~

我们借助 Sphinx 来构建整个 MegEngine 文档（参考 :ref:`how-to-build-the-doc-locally` ），
而 Sphinx 在根据 Docstring 生成 API 页面时，默认只支持 reStructuredText 语法。
因此我们用到了 `sphinx.ext.napoleon 
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ 插件，
它能够在生成文档前临时将所有的 Google Style 语法解析成 reStructureText 语法。

* 这也意味着我们依旧可以使用 :ref:`restructuredtext` 中提到的各种语法来编辑 Docstring 内容，
  包括 API 之间的交叉引用、超链接、插入图片，甚至加入一些高级的 HTML 视觉样式。
* 但我们也要考虑到习惯直接阅读源码（以及使用 ``help()`` / ``print(*.__doc__)`` 语法）的用户，
  使用过多的衍生语法将打破纯文本样式的约定，降低 Python 源码的阅读体验。

We need Chinese!
~~~~~~~~~~~~~~~~

MegEngine 文档的特点之一是提供了中文 API 翻译，而 Docstring 作为源代码的一部分，当然是用英文撰写的。
因此在 MegEngine 源代码中修改 Docstring 后我们还需要在 Documentation 文档中更新对应 ``.po`` 文件，
Sphinx 在生成文档时会先检索匹配到的文本，接着自动地将原文替换成对应的英文，这和 WordPress 国际化原理类似，
相关细节和翻译流程请参考 :ref:`translation` 。


