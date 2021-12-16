.. _cv-examples:

=================================
cv 算法示例
=================================

为了方便大家使用 MegEngine Lite 快速部署 CV 相关算法，MegEngine Lite 提供了一些基本的 CV 算法的部署 demo，为用户在实际部署这些算法
时候提供参考，这些 demo 都是 C++ 编写的，位于 `cv_example <https://github.com/MegEngine/MegEngine/tree/master/lite/example/cpp_example/mge/cv>`_ 中。
编译 :ref:`build-megengine-lite` 的时候将会自动编译 lite_examples。

NN 分类算法
--------------------
NN 分类将从用户读取输入图片，将图片进行预处理之后输入到模型中进行推理，最终输出输入图片中物体的类别信息。代码在 
`picture-classification <https://github.com/MegEngine/MegEngine/tree/master/lite/example/cpp_example/mge/cv/picture_classification.cpp>`_ 。

编译之后运行：lite_examples picture_classification <model.mge> <xxx.jpg>，将看到图片的类别信息。


NN 目标检测
--------------------
部署 `YOLOX <https://github.com/MegEngine/YOLOX>`_ 工程中训练完成的模型。代码在 
`detect-yolox <https://github.com/MegEngine/MegEngine/tree/master/lite/example/cpp_example/mge/cv/detect_yolox.cpp>`_ 。主要代码功能为：

* 输入图片前处理，包括resize，padding，归一化
* 使用 MegEngine Lite 进行推理
* 对输出数据处理

    * 生成对应的框
    * 结合模型的输出数据，选择满足条件的框
    * 对选择的框进行非极大值抑制
    * 输出最终的框

编译之后运行：lite_examples detect_yolox yolox_s.mge <xxx。jpg>，将以 log 形式输出图片中框的信息。其中 yolox_s.mge 来自
`YOLOX <https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.mge>`_ 。
