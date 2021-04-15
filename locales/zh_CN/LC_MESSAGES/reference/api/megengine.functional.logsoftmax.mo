��          �               �   $   �   �     	   �  G   �     �  �     H   �  >   �          #     C     J  g  W  $   �  �   �  	   ~  G   �     �  �   �  H   _  >   �     �     �             :py:class:`~megengine.tensor.Tensor` Applies the :math:`\log(\text{softmax}(x))` function to an n-dimensional input tensor. The :math:`\text{logsoftmax}(x)` formulation can be simplified as: Examples: For numerical stability the implementation follows this transformation: Outputs: \text{logsoftmax}(x)
= \log (\frac{\exp (x)}{\sum_{i}(\exp (x_{i}))})
= x - \log (\sum_{i}(\exp (x_{i})))
= x - \text{logsumexp}(x)

 \text{logsoftmax}(x_{i}) = \log(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} )

 axis along which :math:`\text{logsoftmax}(x)` will be applied. input tensor. megengine.functional.logsoftmax 参数 返回类型 Project-Id-Version:  megengine
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
 :py:class:`~megengine.tensor.Tensor` Applies the :math:`\log(\text{softmax}(x))` function to an n-dimensional input tensor. The :math:`\text{logsoftmax}(x)` formulation can be simplified as: Examples: For numerical stability the implementation follows this transformation: Outputs: \text{logsoftmax}(x)
= \log (\frac{\exp (x)}{\sum_{i}(\exp (x_{i}))})
= x - \log (\sum_{i}(\exp (x_{i})))
= x - \text{logsumexp}(x)

 \text{logsoftmax}(x_{i}) = \log(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} )

 axis along which :math:`\text{logsoftmax}(x)` will be applied. input tensor. megengine.functional.logsoftmax 参数 返回类型 