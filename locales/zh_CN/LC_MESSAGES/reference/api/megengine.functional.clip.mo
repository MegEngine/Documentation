��          �               �   $   �   w   "  	   �     �     �  *   �     �        *     �   B     	            g  $  $   �  �   �     2     9     @     M     i     �     �  �   �     �     �     �   :py:class:`~megengine.tensor.Tensor` Clamps all elements in input tensor into the range `[` :attr:`lower`, :attr:`upper` `]` and returns a resulting tensor: Examples: Outputs: input tensor. lower-bound of the range to be clamped to. megengine.functional.clip output clamped tensor. upper-bound of the range to be clamped to. y_i = \begin{cases}
    \text{lower} & \text{if } x_i < \text{lower} \\
    x_i & \text{if } \text{lower} \leq x_i \leq \text{upper} \\
    \text{upper} & \text{if } x_i > \text{upper}
\end{cases}

 参数 返回 返回类型 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:43+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :py:class:`~megengine.tensor.Tensor` 将输入张量中的所有元素限制（钳位）在 `[` :attr:`lower`, :attr:`upper` `]` 范围内，并返回结果张量： 示例 输出 输入张量 要限制的范围的下限 megengine.functional.clip 被限制后的输出张量 要限制的范围的上限 y_i = \begin{cases}
    \text{lower} & \text{if } x_i < \text{lower} \\
    x_i & \text{if } \text{lower} \leq x_i \leq \text{upper} \\
    \text{upper} & \text{if } x_i > \text{upper}
\end{cases}

 参数 返回 返回类型 