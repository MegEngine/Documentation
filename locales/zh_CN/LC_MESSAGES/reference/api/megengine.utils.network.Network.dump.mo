��    (      \              �     �  %   �  /   �          #  �   =     �  I   �  >        \     l          �     �     �     �     �  S   �  �   ;     �  !   �  �   �  $   �  .   �  H   �     0  Z   P  .   �  S   �  d   .  S   �  A   �  Q   )	  E   {	  I   �	  �   
  >   �
  �   �
     �  g  �     �  %     /   9     i     {  �   �       I   +  >   u     �     �     �     �     �               .  S   ?  �   �       !   ,  �   N  $   �  .     H   ?     �  Z   �  .     S   2  d   �  S   �  A   ?  Q   �  E   �  I     �   c  >   �  �   2     �   0: none of the names are kept 1: (default)keep names of output vars 2: keep names of all (output and internal) vars Keyword Arguments Serializes graph to file. a string for path or a file handler. if is not None, then the dump information for code strip would be written to ``strip_info_file`` enable_chwn4 -- enable_fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty enable_fuse_conv_bias_with_z: whether to fuse conv_bias with z enable_hwcd4 -- enable_io16xc32 -- enable_ioc16 -- enable_nchw32 -- enable_nchw4 -- enable_nchw44 -- enable_nchw44_dot -- enable_nchw88 -- enbale optmizations, will skip all optimize options if this is False. Default: True input for inference on nvidia backend(this optimization pass will result in mismatch of the precision of output of training and inference) into one opr. level for keeping variable names: level for keeping variable names:  * 0: none of the names are kept * 1: (default)keep names of output vars * 2: keep names of all (output and internal) vars megengine.utils.network.Network.dump output file, could be file object or filename. whether output is appended to ``file``. Only works when ``file`` is str. whether to keep operator names. whether to keep param names, so param values can be easily manipulated after loading model whether to keep priority setting for operators whether to use CHWN4 data layout, currently used in nvidia backend with tensorcore. whether to use NCHW32 data layout, currently used in nvidia backend with tensorcore(based on cudnn). whether to use NCHW4 data layout, currently used in nvidia backend(based on cudnn). whether to use NCHW44 data layout, currently used in arm backend. whether to use NCHW44_dot data layout, currently used in armv8.2+dotprod backend. whether to use NCHW88 data layout, currently used in X86 AVX backend. whether to use NHWCD4 data layout. This is faster on some OpenCL backend. whether to use float16 for I/O between oprs and use float32 as internal computation precision. Note the output var would be changed to float16. whether to use float16 for both I/O and computation precision. will be check when `strip_info_file` is not None. if set true, the information for code strip will be append to strip_info_file. if set false, will rewrite strip_info_file 参数 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:46+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 0: none of the names are kept 1: (default)keep names of output vars 2: keep names of all (output and internal) vars Keyword Arguments Serializes graph to file. a string for path or a file handler. if is not None, then the dump information for code strip would be written to ``strip_info_file`` enable_chwn4 -- enable_fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty enable_fuse_conv_bias_with_z: whether to fuse conv_bias with z enable_hwcd4 -- enable_io16xc32 -- enable_ioc16 -- enable_nchw32 -- enable_nchw4 -- enable_nchw44 -- enable_nchw44_dot -- enable_nchw88 -- enbale optmizations, will skip all optimize options if this is False. Default: True input for inference on nvidia backend(this optimization pass will result in mismatch of the precision of output of training and inference) into one opr. level for keeping variable names: level for keeping variable names:  * 0: none of the names are kept * 1: (default)keep names of output vars * 2: keep names of all (output and internal) vars megengine.utils.network.Network.dump output file, could be file object or filename. whether output is appended to ``file``. Only works when ``file`` is str. whether to keep operator names. whether to keep param names, so param values can be easily manipulated after loading model whether to keep priority setting for operators whether to use CHWN4 data layout, currently used in nvidia backend with tensorcore. whether to use NCHW32 data layout, currently used in nvidia backend with tensorcore(based on cudnn). whether to use NCHW4 data layout, currently used in nvidia backend(based on cudnn). whether to use NCHW44 data layout, currently used in arm backend. whether to use NCHW44_dot data layout, currently used in armv8.2+dotprod backend. whether to use NCHW88 data layout, currently used in X86 AVX backend. whether to use NHWCD4 data layout. This is faster on some OpenCL backend. whether to use float16 for I/O between oprs and use float32 as internal computation precision. Note the output var would be changed to float16. whether to use float16 for both I/O and computation precision. will be check when `strip_info_file` is not None. if set true, the information for code strip will be append to strip_info_file. if set false, will rewrite strip_info_file 参数 