# 天元 MegEngine 教程

<p align="center">
  <img height="128" src="./assets/logo/megengine.jpg">
</p>

通过交互式实践的方式学习「[天元 MegEngine](https://github.com/MegEngine/MegEngine) 」深度学习框架的使用和开发。

:school_satchel: [新手入门篇](#新手入门篇) ｜ :rainbow: [模型结构篇](#模型结构篇) ｜ :star: [任务实战篇](#任务实战篇) ｜ :hammer: [生产部署篇](#生产部署篇) 
｜:notebook: [用户手册](#用户手册) ｜ :books: [开发者手册](#开发者手册) 

欢迎在 [天元 MegEngine 论坛](https://discuss.megengine.org.cn/)（我们的中文社区）进行交流反馈～

------

## 新手入门篇

适合从未使用过天元 MegEngine 深度学习框架，以及希望通过代码实践更好地理解深度学习理论的初学者。

- [天元 MegEngine 零基础入门](./notebooks/getting-started/megengine-basic-concepts.ipynb)
- [一个稍微复杂些的线性回归模型](./notebooks/getting-started/learning-from-linear-regression.ipynb)
- [从线性回归到线性分类](./notebooks/getting-started/from-linear-regression-to-linear-classification.ipynb)

## 模型结构篇

可以帮助初涉深度学习领域的研究者快速了解经典模型结构，同时也能够提升相应的编程能力。

- 如何使用 Paper With Code 与 MegEngine Hub 复现、分享和复用模型
- 卷积神经网络（Convolutional Neural Network, CNN）
  - AlexNet
  - VGGNet
  - GoogLeNet/Inception
  - ResNet
  - DenseNet
  - SqueezeNet
  - MobileNet
  - ShuffleNet
  - NasNet
  - EfficientNet
- 循环神经网络（Recurrent Neural Network, RNN）
- 生成对抗网络（Generative Adversarial Network, GAN）
- Transformer

## 任务实战篇

为开源社区提供有趣的任务样例，有助于开发者做出有趣的实际应用。

- 如何使用 Paper With Code 与 MegEngine Hub 复现、分享和复用模型
- 计算机视觉（Computer Vision）
  - 图像分类（Image Classification）
  - 目标检测（Object Detection）
  - 语义分割（Semantic Segmentation）
  - 图像生成（Image Generation）
  - 超分辨率（Super-Resolution）
  - 姿态估计（Pose Estimation）
- 自然语言处理（Natural Language Processing）
  - 机器翻译（Machine Translation）
  - 语言建模（Language Modelling）
  - 问答系统（Quesiton Answering）
  - 情感分析（Sentiment Analysis）
  - 文本生成（Text Generation）
  - 文本摘要（Text Summarization）
- 更多内容欢迎贡献

## 生产部署篇

帮助开发者将模型部署到实际的生产环境中。

## 用户手册

全面了解 MegEngine 的各种使用姿势。

- MegEngine 的安装与环境配置
- MegEngine 使用流程概览
- MegEngine 模型建立与训练
  - 并行分布式训练
  - 量化训练
- MegEngine 模型保存与加载
- MegEngine 模块详解
  - Tensor
  - Functional
  - GradManager
  - Optimizer
- MegEngine 模型优化与性能调试
  - 超参数调整
  - 如何使用 load_and_run
  - 如何使用 Profiler
- MegEngine 模型部署与优化
  - 利用 midout 在端上裁剪

## 开发者手册

加入 MegEngine 的研发队伍，让 MegEngine 变得更好用。

## 版权声明

- Tutorial 中的内容采用 [署名-相同方式共享 4.0 国际](./LICENSE) 许可协议，欢迎合理分发；
- Tutorial 中的代码（如 [examples](./examples) 目录下源代码）默认采用 BSD 许可协议；
- [MegEngine](https://github.com/MegEngine/MegEngine) 采用 [Apache 2.0](https://github.com/MegEngine/MegEngine/blob/master/LICENSE) 许可协议。

## 致谢

- 感谢 [Project Jupyter](https://jupyter.org/), Notebook 使得我们能够以如此灵活的方式进行学习和交流
- 感谢 [ML Visuals](https://github.com/dair-ai/ml-visuals)，我们使用它制作了大部分的可视化图形
- 感谢 [天元 MegEngine 论坛](https://discuss.megengine.org.cn/) 的小伙伴们提供了非常多宝贵建议
- 感谢 [MegStudio](https://studio.brainpp.com/) 提供交互式编程环境和云算力平台支持

