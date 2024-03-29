.. _megengine-document-style-guide:

======================
MegEngine 文档风格指南
======================

.. panels::

   .. image:: ../../../_static/images/bad-document.jpg
      :align: center
   ---

   你的产品有多好并不重要。因为如果配上了一份极其敷衍的文档（Documentation），人们就不会使用它。
   这话听起来可能有些绝对，毕竟在别无选择的时候，用户或许会试着使用你的产品来解决他们的燃眉之急。
   但这种情况下，很难假设用户会按照预期去使用你的产品，要求做到高效率的使用，则更是一种奢望。

   几乎每个人都明白这一点。几乎每个人都知道他们需要好的文档，而且大多数人都试图创建好的文档 ——

   **但大多数人都失败了。/ And most people fail.**

   我们希望提供一个系统的视角，帮助你通过更正确（而不是更努力）的方式来改善你的文档。
   正确的做法往往出人意料地简单 —— 写起来简单，维护起来也简单。

有关文档的“秘密”
----------------

.. note::

   这世上没有一种直接叫做文档的东西，实际上它是以下四个部分——

   * 教程（Tutorials）
   * 操作指南（How-to guides）
   * 技术参考（Reference）
   * 解释（Explaination）

.. warning::

   每类文档都有着其专属的写作方式。 

用户在不同的时间、不同的情景下，可能仅需要阅读其中的某一种材料。
为了做好万全的准备，我们应当将这四种类型的内容全部集成到文档中去，且这四部分需要被明显地结构化，
彼此之间保持独立，各有各的不同。

.. figure:: ../../../_static/images/document-overview.png
   :align: center

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - 
     - Tutorials
     - How-to guides
     - Reference
     - Explanation
   * - 目的是...
     - 帮助学习
     - 达成目标
     - 获取信息
     - 理解背景或细节
   * - 必须能够...
     - 允许刚接触的人快速上手
     - 展示如何解决特定的问题
     - 描述机制
     - 解释清楚
   * - 形式上...
     - 一堂课程
     - 一系列的步骤
     - 术语性质的描述
     - 比较口语的解释
   * - 举例说明...
     - 教小孩子如何做饭
     - 烹饪书中的食谱
     - 烹饪百科全书
     - 一篇关于烹饪社会史的文章

我们可以将它们根据彼此之间的联系和区别，放置在四个象限：

* Tutorials 和 How-to guides 关注实践，而 Reference 和 Explanation 关注理论；
* Tutorials 和 Explanation 服务于学习阶段，而 How-to guides 和 Reference 服务于使用阶段。

这样的划分让作者和读者都能明确材料的性质、材料的种类，以及去哪里获取它们。
它告诉作者如何去写、写什么样的内容，以及写在哪里。同时为作者节省了大量的时间，
避免将时间耗费在试图将想要传递的信息转化成有意义的形式，却由于不知从而下手而纠结不已。

.. note::

   每种类型的文档实际上都只在做一件事情，专注于这件事情，就能达到你的目的。

.. warning::

   实际上，想要维护一份结构不清晰的文档是极其困难的。

   * 每种类型的需求都与其它类型的需求不同，如果文档没能保持上述结构，
     则其中的材料将同时被拉向不同的方向，这对作者和读者而言都是一场灾难。
   * 在 MegEngine 的文档中，并没有独立出解释（Explanation）性质的文档。
     通常它们会按照各自的分类出现在相应教程或指南的恰当位置。
     比如在用户开始按照说明进行操作之前，简单介绍相关背景和概念；
     亦或者是对于某些特定情况下的使用进行具体细节层面的解释。
     这样的安排可以使得用户根据自己的需求去判断是阅读还是略过它们。
   * 不同类型的文档之间出现引用是再正常不过的事情，一千个读者眼中有一千个哈姆雷特，
     此时最困难的事情是如何将这些材料组织，进行自然地串联。

在以下各个小节中，将对这四类材料分别进行详细的介绍。

现在开始写文档吧
----------------

.. toctree::
   :maxdepth: 1

   tutorials
   how-to-guides
   reference
   explanation

.. admonition:: 对于作者
   :class: note

   文档维护人员必须处理的大难题之一是：要清楚地了解现在应该做什么。
   否则容易出现这种情况 —— 尽管写了很多遍改了很多遍，但发现很难以令人满意的方式将已有内容组合在一起。

   我们可以借助文档系统的概念，通过明确区分不同的材料来解决上述问题，
   每种材料都有自己的写作技巧，这使得文档更容易编写、维护和使用。
   但好的文档不是从从石头里蹦出来的 —— 现在就可以尝试开始编写它，
   不必去过于担心应该写哪些内容，或采用何种写作风格。

.. admonition:: 对于读者
   :class: note

   文档系统能够更好地为用户服务，因为他们在软件交互周期的所有不同的阶段，
   都能够找到适合当时需求的正确类型的文档材料。

   编写用途明确的文档，把它们放在正确的象限，有助于帮助软件吸引和留住更多用户，
   他们将更有效地使用软件来解决自己需求 —— 而这才是软件开发者梦寐以求的画面。

致谢
----

.. note::

   《MegEngine 文档风格指南》的撰写受到了 `Documentation system <https://documentation.divio.com/>`_ 的启发，
   其中部分内容根据项目自身情况而有所删改，同时也针对 MegEngine 文档中的一些情景进行了拓展。
