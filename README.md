# MegEngine 官方文档

<p align="center">
  <img height="128" src="./source/_static/logo/megengine.jpg">
</p>

[![language-zh](https://img.shields.io/badge/language-zh-brightgreen)](https://megengine.org.cn/doc/stable/zh/) [![language-en](https://img.shields.io/badge/language-en-brightgreen)](https://megengine.org.cn/doc/stable/en/) [![Crowdin](https://badges.crowdin.net/megengine/localized.svg)](https://crowdin.com/project/megengine) [![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

根据官方文档学习旷视天元「[MegEngine](https://github.com/MegEngine/MegEngine) 」深度学习框架的使用和开发。

欢迎在 [MegEngine 论坛](https://discuss.megengine.org.cn/)（我们的中文社区）进行交流反馈～ 

:point_right: 你也可以选择 [成为文档贡献者](./CONTRIBUTING.md) 或者 [帮助我们进行翻译](https://crowdin.com/project/megengine) 。

## 如何在本地构建与预览文档

参考 [这个文件](./source/development/contribute-to-docs/build-the-doc-locally.rst) 。

## 分支名解释

MegEngine 文档的多分支维护逻辑与 MegEngine 版本发布逻辑不同：

- 当某个 MegEngine 版本正式对外发布时，对应版本的文档也会发布在文档官网；
- ``main`` 分支用于在当前已经发布的 stable 版本做内容的补充或修复；
- ``dev`` 分支用于在未发布版本，或者是 rc 版本做内容的提前更新。

|  分支名  | 对应版本  |  对应 MegEngine 分支  |
|  ------  | --------  |  -------------------  |
|   dev    |   1.7     |        master         |
|   main   |   1.6     |      release-1.6      |

在新的 stable 版本发布后，通常会将 dev 分支的变动 rebase 到 main 分支。

注： 旧版本的文档已经被归档，里面的内容不会再进行更新。


