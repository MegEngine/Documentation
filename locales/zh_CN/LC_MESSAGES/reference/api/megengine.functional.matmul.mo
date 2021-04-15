��          �               \  $   ]  +   �  .   �  	   �  �   �  9   �     �  G   �  =   D     �  B   �  6   �  9        U     q     �     �     �     �  g  �  $   #  +   H  .   t  	   �  �   �  9        �  G   �  =   
     H  B   g  6   �  9   �     	     7	     F	     f	     m	     t	   :py:class:`~megengine.tensor.Tensor` Both 1-D tensor, simply forward to ``dot``. Both 2-D tensor, normal matrix multiplication. Examples: If at least one tensor are 3-dimensional or >3-dimensional, the other tensor should have dim >= 2, the batched matrix-matrix is returned, and the tensor with smaller dimension will be broadcasted. For example: If one input tensor is 1-D, matrix vector multiplication. Outputs: Performs a matrix multiplication of the matrices ``inp1`` and ``inp2``. With different inputs dim, this function behaves differently: first matrix to be multiplied. inp1: `(n, j, k, m)`, inp2: `(n, j, m, p)`, return: `(n, j, k, p)` inp1: `(n, k, m)`, inp2: `(m, p)`, return: `(n, k, p)` inp1: `(n, k, m)`, inp2: `(n, m, p)`, return: `(n, k, p)` megengine.functional.matmul output tensor. second matrix to be multiplied. 参数 返回 返回类型 Project-Id-Version:  megengine
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
 :py:class:`~megengine.tensor.Tensor` Both 1-D tensor, simply forward to ``dot``. Both 2-D tensor, normal matrix multiplication. Examples: If at least one tensor are 3-dimensional or >3-dimensional, the other tensor should have dim >= 2, the batched matrix-matrix is returned, and the tensor with smaller dimension will be broadcasted. For example: If one input tensor is 1-D, matrix vector multiplication. Outputs: Performs a matrix multiplication of the matrices ``inp1`` and ``inp2``. With different inputs dim, this function behaves differently: first matrix to be multiplied. inp1: `(n, j, k, m)`, inp2: `(n, j, m, p)`, return: `(n, j, k, p)` inp1: `(n, k, m)`, inp2: `(m, p)`, return: `(n, k, p)` inp1: `(n, k, m)`, inp2: `(n, m, p)`, return: `(n, k, p)` megengine.functional.matmul output tensor. second matrix to be multiplied. 参数 返回 返回类型 