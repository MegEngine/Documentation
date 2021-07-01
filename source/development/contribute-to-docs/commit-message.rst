.. _commit-message:

=======================
Commit message 书写指南
=======================

.. note::

   * 该指南中的规则不一定适用于除 MegEngine 文档以外的项目。
   * 大部分情况下允许使用精简写法的 Commit Message.


标准写法
--------

一般而言，每个 commit 都有对应至少一个 Issue/Pull Request 与之相关联。

标准的 commit message 包含 Header、Body 和 Footer 三个部分（彼此之间空一行）：

.. code-block:: shell

   <type>(<scope>): <short summary>
   // 空一行
   <body>
   // 空一行
   <footer>

模版如下：

.. code-block:: 

   type(scope): summarize changes in around 50 characters or less

   More detailed explanatory text, if necessary. Wrap it to about 72
   characters or so. In some contexts, the first line is treated as the
   subject of the commit and the rest of the text as the body. The
   blank line separating the summary from the body is critical (unless
   you omit the body entirely); various tools like `log`, `shortlog`
   and `rebase` can get confused if you run the two together.

   Explain the problem that this commit is solving. Focus on why you
   are making this change as opposed to how (the code explains that).
   Are there side effects or other unintuitive consequences of this
   change? Here's the place to explain them.

   Further paragraphs come after blank lines.

   - Bullet points are okay, too
   - Typically a hyphen or asterisk is used for the bullet, preceded
     by a single space, with blank lines in between, but conventions
     vary here

   If you use an issue tracker, put references to them at the bottom,
   like this:

   Resolves: #123
   See also: #456, #789

参考样例： `33aaf43 <https://github.com/MegEngine/Documentation/commit/33aaf430848be409ab46e19733be40a3bfc6abb8>`_

Header
~~~~~~

Header 通常也称为 Subject，是查看 ``git log`` 时必需的信息：

.. code-block:: shell

   <type>(<scope>): <short summary>
 
其中类型（Type）主要有以下几类：

* 文档（docs）：**最主要也是最常见的类型，** 所有文档内容的增删查改都归于此类
* 样式（style）：对文档格式的修改，通常是为了让内容看起来更清晰美观
* 重构（refactor）：对文档内容的结构性调整，可以是多个文件结构的重构
* 翻译（trans）：翻译英文 API docstring 或者将中文文档翻译成英文
* 构建（build）：Sphinx 引擎构建文档逻辑流程、配置文件相关的改动
* 持续集成（ci）：在这个项目中专指 GitHub Actions 中的一些工作流改动
* 特性（feat）：专指文档功能特性的变化，比如使用新的 Sphinx 插件等等
* 修复（fix）：对导致文档无法正常显示的一类 Bug 的修复

范围（Scope）是可选项，根据修改所影响的内容而定，常见类型有：

* 模块名： ``data``, ``tensor``, ``functional`` 等
* 所属分类： ``tutorial``, ``guide``, ``example`` 等

总结（Summary）是对 commit 的简短描述，要求如下：

* 不超过 50 个字符
* 动词开头，使用第一人称现在时，比如 change 而不是 changes 或 changed
* 第一个字母小写
* 结尾不加句号

Body
~~~~

当需要对 commit 进行更加详细的描述时，通常会将其放在正文部分。
更常见的情况是，在 Issue/Pull Request 中进行具体的讨论和更改，
仅在有必要的情况下，会选择在 commit message 中说明原因和影响。

Footer
~~~~~~

用于添加各种参考信息，比如 Issues/Pull Request 的 ID，或参考的网页链接等等。
由于 MegEngine 文档使用 GitHub 处理工作流，因此也可以参考
`Linking a pull request to an issue using a keyword 
<https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue>`_ . 

精简写法
--------

当 Commit message 中不包含 Body 部分时，可进行精简表示:

.. code-block:: shell

   docs(tutorial): correct the xxx typo (#123)

其中 # 后跟着的数字是对应的 Issue/Pull Request ID.

