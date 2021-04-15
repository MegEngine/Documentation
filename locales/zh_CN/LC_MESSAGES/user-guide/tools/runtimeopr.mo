��          �               �   #   �      !  �   9  a     �   z     -  n   F  �   �  S   a     �      �  �   �  9   �  �  %  #   �     �  �   �  a   �  �   ,	     �	  n   �	  �   g
  S        g      �  �   �  9   �   RuntimeOpr 作为模型的一部分 RuntimeOpr 使用说明 RuntimeOpr 指通过 MegEngine 将其它硬件厂商支持的离线模型作为一个算子嵌入到 MegEngine Graph 中， 进而方便地使用 MegEngine 提供的各类 :ref:`图手术 <graphsurgeon>` 和推理工具。 RuntimeOpr 输入的所属设备应该是该类设备，本例中 inp 的 device 为 “atlas0” 包含 RuntimeOpr 的模型无法通过 :py:func:`megengine.save` 保存权重， 只能通过 :py:meth:`.trace.dump` 直接保存为模型。用法见 :ref:`runtimeopr-dump` 。 参考下面的代码： 只能从 CPU 拷贝到其他设备或者反之，各类设备之间不能直接拷贝，比如 GPU 到 Atlas. 在 RuntimeOpr 前后必须使用 :py:func:`~.copy` 把 Tensor 从 CPU 拷贝到 Atlas, 或者从 Atlas 拷贝到 CPU, 不然会因为 CompNode 不符合规范而报错； 如果需要转变数据类型，请在 CPU 上完成（参考上面的代码）； 序列化与反序列化 模型只包含一个 RuntimeOpr 目前支持 RuntimeOpr 的类型有 TensorRT、Atlas 和 Cambricon 三种， 包含 RuntimeOpr 的模型需要在对应的硬件平台上才能执行推理任务。 下面以 Atlas 为例展示用法（TensorRT、Cambricon 的接口与之类似）： 硬件厂商的模型文件需要以字节流形式打开 Project-Id-Version: MegEngine 1.3.0
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-09 17:59+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_Hans_CN
Language-Team: zh_Hans_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 RuntimeOpr 作为模型的一部分 RuntimeOpr 使用说明 RuntimeOpr 指通过 MegEngine 将其它硬件厂商支持的离线模型作为一个算子嵌入到 MegEngine Graph 中， 进而方便地使用 MegEngine 提供的各类 :ref:`图手术 <graphsurgeon>` 和推理工具。 RuntimeOpr 输入的所属设备应该是该类设备，本例中 inp 的 device 为 “atlas0” 包含 RuntimeOpr 的模型无法通过 :py:func:`megengine.save` 保存权重， 只能通过 :py:meth:`.trace.dump` 直接保存为模型。用法见 :ref:`runtimeopr-dump` 。 参考下面的代码： 只能从 CPU 拷贝到其他设备或者反之，各类设备之间不能直接拷贝，比如 GPU 到 Atlas. 在 RuntimeOpr 前后必须使用 :py:func:`~.copy` 把 Tensor 从 CPU 拷贝到 Atlas, 或者从 Atlas 拷贝到 CPU, 不然会因为 CompNode 不符合规范而报错； 如果需要转变数据类型，请在 CPU 上完成（参考上面的代码）； 序列化与反序列化 模型只包含一个 RuntimeOpr 目前支持 RuntimeOpr 的类型有 TensorRT、Atlas 和 Cambricon 三种， 包含 RuntimeOpr 的模型需要在对应的硬件平台上才能执行推理任务。 下面以 Atlas 为例展示用法（TensorRT、Cambricon 的接口与之类似）： 硬件厂商的模型文件需要以字节流形式打开 