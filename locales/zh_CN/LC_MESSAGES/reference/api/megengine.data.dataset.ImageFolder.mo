��          �               L  n   M  M   �  Q   
     \     u  Q   �     �  ;   �     -  H   5  �   ~  q     6   �  "   �  "   �     �       I   
  g  T  n   �  M   +  Q   y     �     �  Q   �     O  ;   `     �  H   �  �   �  q   }	  6   �	  "   &
  "   I
     l
     r
  I   y
   :obj:`__init__ <megengine.data.dataset.ImageFolder.__init__>`\ \(root\[\, check\_valid\_func\, class\_name\]\) :obj:`collect_class <megengine.data.dataset.ImageFolder.collect_class>`\ \(\) :obj:`collect_samples <megengine.data.dataset.ImageFolder.collect_samples>`\ \(\) :py:class:`~typing.Dict` :py:class:`~typing.List` ImageFolder is a class for loading image data and labels from a organized folder. Initialize self. Labels are indices of sorted classes in the root directory. Methods The folder is expected to be organized as followed: root/cls/xxx.img_ext a function used to check if files in folder are expected image files, if ``None``, default function that checks file extensions will be called. a function used to load image from path, if ``None``, default function that loads images with PIL will be called. if ``True``, return class name instead of class index. megengine.data.dataset.ImageFolder root directory of an image folder. rtype 参数 基类：:class:`megengine.data.dataset.vision.meta_vision.VisionDataset` Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:41+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :obj:`__init__ <megengine.data.dataset.ImageFolder.__init__>`\ \(root\[\, check\_valid\_func\, class\_name\]\) :obj:`collect_class <megengine.data.dataset.ImageFolder.collect_class>`\ \(\) :obj:`collect_samples <megengine.data.dataset.ImageFolder.collect_samples>`\ \(\) :py:class:`~typing.Dict` :py:class:`~typing.List` ImageFolder is a class for loading image data and labels from a organized folder. Initialize self. Labels are indices of sorted classes in the root directory. Methods The folder is expected to be organized as followed: root/cls/xxx.img_ext a function used to check if files in folder are expected image files, if ``None``, default function that checks file extensions will be called. a function used to load image from path, if ``None``, default function that loads images with PIL will be called. if ``True``, return class name instead of class index. megengine.data.dataset.ImageFolder root directory of an image folder. rtype 参数 基类：:class:`megengine.data.dataset.vision.meta_vision.VisionDataset` 