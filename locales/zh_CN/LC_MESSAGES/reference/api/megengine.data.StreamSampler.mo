��    	      d               �   L   �   �   �      �     �     �  �   �     {  0   �  g  �  L   1  �   ~     =     N     V  �   r     �  0      :obj:`__init__ <megengine.data.StreamSampler.__init__>`\ \(\[batch\_size\]\) In the case of multiple machines, sampler should ensure that each worker gets different data. But this class cannot do it yet, please build your own dataset and sampler to achieve this goal. Initialize self. Methods Sampler for stream dataset. Usually, meth::`~.StreamDataset.__iter__` can return different iterator by ``rank = dist.get_rank()``. So that they will get different data. megengine.data.StreamSampler 基类：:class:`megengine.data.sampler.Sampler` Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:42+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :obj:`__init__ <megengine.data.StreamSampler.__init__>`\ \(\[batch\_size\]\) In the case of multiple machines, sampler should ensure that each worker gets different data. But this class cannot do it yet, please build your own dataset and sampler to achieve this goal. Initialize self. Methods Sampler for stream dataset. Usually, meth::`~.StreamDataset.__iter__` can return different iterator by ``rank = dist.get_rank()``. So that they will get different data. megengine.data.StreamSampler 基类：:class:`megengine.data.sampler.Sampler` 