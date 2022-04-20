.. _cpp-doc-style-guide:

==================================
C++ 文档 Doxygen/Breathe 语法
==================================

.. note::

   本手册用来给 MegEngine 开发者提供常见的 C++ 接口文档的语法模板作为参考，
   帮助其高效地撰写和修改符合 MegEngine 工程规范的面向用户的 C++ 接口文档。

.. warning:: 

   不是所有的 Doxygen 语法（如折叠等）都能够在 MegEngine 文档中被使用，
   我们仅使用它来提取信息，而用 Breathe 来构建文档，参考 :ref:`cpp-doc-process` 小节。

一份参考模板
-------------------

下面这个例子修改自 Breathe 官方文档的 `Nutshell <https://breathe.readthedocs.io/en/latest/index.html#in-a-nutshell>`_ 示例：

.. code-block:: cpp

   /**
    * @file nutshell.h
    * 
    * @brief This is a nutshell example.
    */

   /*!
    *  With a little bit of a elaboration, should you feel it necessary.
    */
   class Nutshell
   {
   public:

      //! Our tool set
      /*! The various tools we can opt to use to crack this particular nut */
      enum Tool
      {
         kHammer = 0,          //!< What? It does the job
         kNutCrackers,         //!< Boring
         kNinjaThrowingStars   //!< Stealthy
      };

      //! Nutshell constructor
      Nutshell();

      //! Nutshell destructor
      ~Nutshell();

      /*! Crack that shell with specified tool
       *
       * @param tool - the tool with which to crack the nut
       */
      void crack( Tool tool );

      /*!
       * @return Whether or not the nut is cracked
       */
      bool isCracked();

   private:

      //! Our cracked state
      bool m_isCracked;

   };

.. admonition:: 注意事项
   :class: warning

   * 多行注释中带有前导星号（leading-asterisk） ``*``, 而非单纯的空行；
   * 使用 ``@param`` 代替 ``\param`` 写法，避免遇到后者形式在其它解析工具中被转义的情况；
   * MegEngine 文档中只要是公开接口、成员，不论是否有相应备注，都会生成对应的 API 文档。

使用 Breathe 语法
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Breathe 文档中提供了一些简单的语法模板，可以在 C++ 接口文档中添加数学公式、列表、表格等样式，在简单情况下可以使用。

参考 `Breathe Features <https://breathe.readthedocs.io/en/latest/index.html#features>`_ 页面中给出的各种样例。

对于比较复杂的文档内容编辑和排版需求，推荐使用下一小节提到的 ReST 语法，即 MegEngine 文档中最常使用的语法。

使用 ReST 语法和组件
~~~~~~~~~~~~~~~~~~~~~~~~~~

MegEngine 文档中使用了比较多的 Sphinx 拓展样式，使用时通过 ReStructuredtext 语法来解析。
当默认的样式不满足需求时，可以使用 :ref:`restructuredtext` 中展示的各种语法作为拓展。
但需要注意的是，由于使用了前导星号，为了能够被正常解析，需要使用 ``embed:rst:leading-asterisk`` 标记：

.. code-block:: rst
   :emphasize-lines: 4, 8

   /*!
   * Inserting additional reStructuredText information.
   *
   * \verbatim embed:rst:leading-asterisk
   *     .. note::
   *
   *        This is a example.
   * \endverbatim
   */

它等同于在 C++ 接口文档中插入了如下 ReST 语法：

.. code-block:: rst

   .. note::
   
      This is a example.

会得到对应的 ``note`` 样式块内容。同理，你还可以使用这种方法来插入 `数学公式 <math-rst>`_ 和图片等等内容。

.. _cpp-doc-process:

从源码到文档的流程
-------------------

MegEngine 的 C++ 源码经历了如下流程变成 MegEngine 文档中的 API 参考页面：

.. mermaid::
   :align: center

   flowchart LR
       HF[C++ Head Files] --> |Doxygen| XML[XML Files]
       XML --> |Breathe Directive| Sphinx[Sphinx RST Doc]

由于 MegEngine 文档与 MegEngine 源码不在同一处维护，
因此开发人员通常会规律性地使用 Doxygen 从 MegEngine 的 C++ 源码中生成最新的 XML 文件
（位于 :docs:`doxyxml` 目录中）。
平时撰写文档只需要使用 Breathe 将 XML 中的信息转换成 Sphinx 的 RST 文档，
体验上与从 MegEngine 的 Python Package 生成 API 文档类似。

以 Tensor 为例子，添加 Python 接口和 C++ 接口（生成文档）的 Sphinx 语法对比如下：

.. code-block:: rst

   .. autoclass:: megengine.Tensor

   .. doxygenclass:: lite::Tensor

使用自动生成的文档的好处之一是，方便在文档其它的任何地方进行引用——

.. code-block:: rst

    比如此处直接引用 :class:`megengine.Tensor` 与 :cpp:class:`lite::Tensor` 的 API 文档。

比如此处直接引用 :class:`megengine.Tensor` 与 :cpp:class:`lite::Tensor` 的 API 文档。

.. admonition:: 详细的 Sphinx 和 Breathe 文档语法对比

   * `sphinx.ext.autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ – Include documentation from docstrings
   * `Breathe Directives & Config Variables <https://breathe.readthedocs.io/en/latest/directives.html>`_ – Breathe directives and config variables
   * `交叉引用 <cross-reference-rst>`_ - 在 MegEngine 文档中引用 API 页面