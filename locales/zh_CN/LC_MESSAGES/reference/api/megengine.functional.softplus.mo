��          |               �   $   �   "     	   %     /  (   8  �   a            �   :     �        g    $   u     �  	   �     �  (   �  �   �     �     �  �   �     �     �   :py:class:`~megengine.tensor.Tensor` Applies the element-wise function: Examples: Outputs: \text{softplus}(x) = \log(1 + \exp(x))

 \text{softplus}(x) = \log(1 + \exp(x))
                   = \log(1 + \exp(-\text{abs}(x))) + \max(x, 0)
                   = \log1p(\exp(-\text{abs}(x))) + \text{relu}(x)

 input tensor. megengine.functional.softplus softplus is a smooth approximation to the ReLU function and can be used to constrain the output to be always positive. For numerical stability the implementation follows this transformation: 参数 返回类型 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:40+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :py:class:`~megengine.tensor.Tensor` 逐元素应用以下函数： Examples: Outputs: \text{softplus}(x) = \log(1 + \exp(x))

 \text{softplus}(x) = \log(1 + \exp(x))
                   = \log(1 + \exp(-\text{abs}(x))) + \max(x, 0)
                   = \log1p(\exp(-\text{abs}(x))) + \text{relu}(x)

 input tensor. megengine.functional.softplus softplus is a smooth approximation to the ReLU function and can be used to constrain the output to be always positive. For numerical stability the implementation follows this transformation: 参数 返回类型 