��          �               �   $   �   	          �   %     �     �  3   �  �   /     �     �     �     �  g  �  $   d  	   �  	   �  �   �     �     �  1   �    �     �     �              :py:class:`~megengine.tensor.Tensor` Examples: Outputs: Returns a new tensor where each of the elements are randomly set to zero with probability P = ``drop_prob``. Optionally rescale the output tensor if ``training`` is True. input tensor. megengine.functional.dropout probability to drop (set to zero) a single element. the default behavior of ``dropout`` during training is to rescale the output, then it can be replaced by an :class:`~.Identity` during inference. Default: True the output tensor 参数 返回 返回类型 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:39+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :py:class:`~megengine.tensor.Tensor` 示例： 输出： 以 ``drop_prob`` 为概率，随机将输入张量中的元素设置成 0，并返回这个新的张量。当 ``training`` 设置为 True 时，输出值将按照概率进行缩放以保证数值的期望与未 dropout 前保持一致 输入张量 megengine.functional.dropout 丢弃掉单个元素（设置成0）的概率。 ``dropout`` 的默认行为会将输出进行缩放，从而在推理时可被等价替换为:class:`~.Identity`，当 ``training`` 设置为 True 时，输出值将按照概率进行缩放以保证数值的期望与未 dropout 前保持一致。默认值：True 输出张量 参数 返回 返回类型 