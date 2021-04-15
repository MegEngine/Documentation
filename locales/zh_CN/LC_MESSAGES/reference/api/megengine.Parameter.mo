��    9      �              �  !   �  _   �  5   /  +   e  0   �  +   �  9   �  )   (  2   R  '   �  2   �  =   �  ?     =   ^  '   �  '   �  .   �  ?     -   [  9   �  8   �  8   �  ;   5  )   q  '   �  =   �  .   	  0   0	  <   a	  =   �	  
   �	  0   �	     
     )
  n   1
  V   �
  M   �
  Z   E  A   �  G   �  ;   *  V   f  W   �  V     ;   l  T   �  /   �  P   -  &   ~  Y   �     �          -  $   F     k  )     g  �  !     _   3  5   �  +   �  0   �  +   &  9   R  )   �  2   �  '   �  2     =   D  ?   �  =   �  '      '   (  .   P  ?     -   �  9   �  8   '  8   `  ;   �  )   �  '   �  =   '  .   e  0   �  <   �  =     
   @  0   K     |     �  n   �  V     M   [  Z   �  A     G   F  ;   �  V   �  W   !  V   y  ;   �  T     /   a  P   �  &   �  Y   	     c     z     �  $   �     �  )   �   :obj:`T <megengine.Parameter.T>`\ :obj:`__init__ <megengine.Parameter.__init__>`\ \(data\[\, dtype\, device\, is\_const\, ...\]\) :obj:`astype <megengine.Parameter.astype>`\ \(dtype\) :obj:`c_name <megengine.Parameter.c_name>`\ :obj:`detach <megengine.Parameter.detach>`\ \(\) :obj:`device <megengine.Parameter.device>`\ :obj:`dmap_callback <megengine.Parameter.dmap_callback>`\ :obj:`dtype <megengine.Parameter.dtype>`\ :obj:`flatten <megengine.Parameter.flatten>`\ \(\) :obj:`grad <megengine.Parameter.grad>`\ :obj:`item <megengine.Parameter.item>`\ \(\*args\) :obj:`max <megengine.Parameter.max>`\ \(\[axis\, keepdims\]\) :obj:`mean <megengine.Parameter.mean>`\ \(\[axis\, keepdims\]\) :obj:`min <megengine.Parameter.min>`\ \(\[axis\, keepdims\]\) :obj:`name <megengine.Parameter.name>`\ :obj:`ndim <megengine.Parameter.ndim>`\ :obj:`numpy <megengine.Parameter.numpy>`\ \(\) :obj:`prod <megengine.Parameter.prod>`\ \(\[axis\, keepdims\]\) :obj:`qparams <megengine.Parameter.qparams>`\ :obj:`requires_grad <megengine.Parameter.requires_grad>`\ :obj:`reset_zero <megengine.Parameter.reset_zero>`\ \(\) :obj:`reshape <megengine.Parameter.reshape>`\ \(\*args\) :obj:`set_value <megengine.Parameter.set_value>`\ \(value\) :obj:`shape <megengine.Parameter.shape>`\ :obj:`size <megengine.Parameter.size>`\ :obj:`sum <megengine.Parameter.sum>`\ \(\[axis\, keepdims\]\) :obj:`to <megengine.Parameter.to>`\ \(device\) :obj:`tolist <megengine.Parameter.tolist>`\ \(\) :obj:`transpose <megengine.Parameter.transpose>`\ \(\*args\) A kind of Tensor that is to be considered a module parameter. Attributes Copy self :class:`~.Tensor` to specified device. Initialize self. Methods Returns a :class:`Tensor` with the same data and number of elements with the specified :attr:`~.Tensor.dtype`. Returns a :class:`numpy.dtype` object represents the data type of a :class:`~.Tensor`. Returns a :class:`tuple` or a :class:`~.Tensor` represents tensor dimensions. Returns a :class:`~.QParams` object containing quantization params of a :class:`~.Tensor`. Returns a new :class:`~.Tensor`, detached from the current graph. Returns a string represents the device a :class:`~.Tensor` storaged on. Returns self :class:`~.Tensor` as a :class:`numpy.ndarray`. Returns the max value of each row of the input tensor in the given dimension ``axis``. Returns the mean value of each row of the input tensor in the given dimension ``axis``. Returns the min value of each row of the input tensor in the given dimension ``axis``. Returns the number of dimensions of self :class:`~.Tensor`. Returns the product of each row of the input tensor in the given dimension ``axis``. Returns the size of the self :class:`~.Tensor`. Returns the sum of each row of the input tensor in the given dimension ``axis``. Returns the tensor as a (nested) list. Returns the value of this :class:`~.Tensor` as a standard Python :class:`numbers.Number`. See :func:`~.flatten`. See :func:`~.reshape`. See :func:`~.transpose`. alias of :attr:`~.Tensor.transpose`. megengine.Parameter 基类：:class:`megengine.tensor.Tensor` Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:31+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :obj:`T <megengine.Parameter.T>`\ :obj:`__init__ <megengine.Parameter.__init__>`\ \(data\[\, dtype\, device\, is\_const\, ...\]\) :obj:`astype <megengine.Parameter.astype>`\ \(dtype\) :obj:`c_name <megengine.Parameter.c_name>`\ :obj:`detach <megengine.Parameter.detach>`\ \(\) :obj:`device <megengine.Parameter.device>`\ :obj:`dmap_callback <megengine.Parameter.dmap_callback>`\ :obj:`dtype <megengine.Parameter.dtype>`\ :obj:`flatten <megengine.Parameter.flatten>`\ \(\) :obj:`grad <megengine.Parameter.grad>`\ :obj:`item <megengine.Parameter.item>`\ \(\*args\) :obj:`max <megengine.Parameter.max>`\ \(\[axis\, keepdims\]\) :obj:`mean <megengine.Parameter.mean>`\ \(\[axis\, keepdims\]\) :obj:`min <megengine.Parameter.min>`\ \(\[axis\, keepdims\]\) :obj:`name <megengine.Parameter.name>`\ :obj:`ndim <megengine.Parameter.ndim>`\ :obj:`numpy <megengine.Parameter.numpy>`\ \(\) :obj:`prod <megengine.Parameter.prod>`\ \(\[axis\, keepdims\]\) :obj:`qparams <megengine.Parameter.qparams>`\ :obj:`requires_grad <megengine.Parameter.requires_grad>`\ :obj:`reset_zero <megengine.Parameter.reset_zero>`\ \(\) :obj:`reshape <megengine.Parameter.reshape>`\ \(\*args\) :obj:`set_value <megengine.Parameter.set_value>`\ \(value\) :obj:`shape <megengine.Parameter.shape>`\ :obj:`size <megengine.Parameter.size>`\ :obj:`sum <megengine.Parameter.sum>`\ \(\[axis\, keepdims\]\) :obj:`to <megengine.Parameter.to>`\ \(device\) :obj:`tolist <megengine.Parameter.tolist>`\ \(\) :obj:`transpose <megengine.Parameter.transpose>`\ \(\*args\) A kind of Tensor that is to be considered a module parameter. Attributes Copy self :class:`~.Tensor` to specified device. Initialize self. Methods Returns a :class:`Tensor` with the same data and number of elements with the specified :attr:`~.Tensor.dtype`. Returns a :class:`numpy.dtype` object represents the data type of a :class:`~.Tensor`. Returns a :class:`tuple` or a :class:`~.Tensor` represents tensor dimensions. Returns a :class:`~.QParams` object containing quantization params of a :class:`~.Tensor`. Returns a new :class:`~.Tensor`, detached from the current graph. Returns a string represents the device a :class:`~.Tensor` storaged on. Returns self :class:`~.Tensor` as a :class:`numpy.ndarray`. Returns the max value of each row of the input tensor in the given dimension ``axis``. Returns the mean value of each row of the input tensor in the given dimension ``axis``. Returns the min value of each row of the input tensor in the given dimension ``axis``. Returns the number of dimensions of self :class:`~.Tensor`. Returns the product of each row of the input tensor in the given dimension ``axis``. Returns the size of the self :class:`~.Tensor`. Returns the sum of each row of the input tensor in the given dimension ``axis``. Returns the tensor as a (nested) list. Returns the value of this :class:`~.Tensor` as a standard Python :class:`numbers.Number`. See :func:`~.flatten`. See :func:`~.reshape`. See :func:`~.transpose`. alias of :attr:`~.Tensor.transpose`. megengine.Parameter 基类：:class:`megengine.tensor.Tensor` 