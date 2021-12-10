.. _model_pack:

================================
模型加解密和打包（可选）
================================

MegEngine Lite 提供的模型打包功能主要包括：

* 模型加密：
    为了保障模型研发的知识产权，防止恶意逆向，剽窃他人成果，MegEngine Lite 中支持对模型加密功能，用户开发完成之后可以对模型进行加密，
    并在模型 Load 时候对加密之后的模型进行解密，该过程中用户可以注册自定义的解密算法，以及使用 MegEngine Lite 提供的解密算法并使用
    自定义的秘钥。

* 模型自解释：
    MegEngine Lite 中包含众多的配置，参考 :ref:`option_config`，这些配置可能会因模型不同而同步，或者平台不同而不同，
    如果将这些配置都必须在用户代码中体现，将增加用户配置模型的负担，也造成集成 MegEngine Lite 时候代码不统一，增加额外的
    用户使用成本，为此 MegEngine Lite 支持将一些必要的配置信息打包到模型中，并在模型载入时候自动完成 Network 的配置，达到
    代码统一。另外打包到模型里面的信息还包括一些模型的基本信息，如：模型的 IO 配置，模型的加密方式等，所以这个功能简称：模型自解释。

模型获取方式参考 :ref:`lite-model-dump`，将获得 resnet50.mge 模型。

加密模型
---------------------

MegEngine Lite 可以加载的原始模型和经过加密算法加密的模型：

* 原始模型：直接将完成训练的模型在 MegEngine 环境中进行 dump 生成的模型。
* 原始加密模型：将上述 dump 的模型通过加密算法进行加密的模型。

MegEngine Lite 提供了三种加密方式，分别是 AES， RC4，simple_fast_RC4，这些加密方式都支持用户设置自己的加解密秘钥，他们对应在 MegEngine Lite
中的名字分别为："AES_default"，"RC4_default"，"SIMPLE_FAST_RC4_default"，后续将可以通过这些字符串对其进行索引。

下面以 resnet50.mge 加密过程作为 example 展示使用 MegEngine Lite 中提供的工具进行模型加密。

AES 加密
^^^^^^^^^^^^^^^^^^^^^^^

MegEngine Lite 中 AES 的加密工具为：:src:`lite/tools/aes_encrypt.sh` 的一个脚本，这个脚本可以将一个文件通过指定的的秘钥加密成一个 ASE 加密的文件，
其中秘钥为 32 个字节 16 进制数，如果不指定具体秘钥，将使用 MegEngine Lite 中预先设置的 **公开** 秘钥。

.. code-block:: bash

    lite/tools/aes_encrypt.sh resnet50.mge resnet50_encrypted.mge 000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F

上面命令将 resnet50.mge 模型加密成为 resnet50_encrypted.mge，并且加密时候使用的秘钥为 000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F，
其中秘钥每两个数字组成 1 个 16 进制数并在内存中占用一个字节，一共需要 32 个这样的 16 进制数组成 AES 的秘钥，上面的秘钥为 32 个从 0 到 31 的 16 进制数组成。

RC4 加密
^^^^^^^^^^^^^^^^^^^^^^^

MegEngine Lite 中支持的 RC4 和 simple_fast_RC4 的加密工具为 C++ 代码编译的一个小工具，具体代码在：:src:`lite/tools/rc4_encypt.cpp` ，
这个工具在 :ref:`build-megengine-lite` 中编译 MegEngine Lite 时默认会编译出来，编译之后为：rc4_encryptor，编译出来保存在 install 目录下的
lite/tools/rc4_encryptor，这个工具可以通过指定的秘钥或者 MegEngine Lite 默认的 **公开** 秘钥加密文件。

.. code-block:: bash

    #使用编译之后的工具加密模型
    ./rc4_encryptor <预定义的加密方法> <加密之前的file> <加密之后的file>
    ./rc4_encryptor <加密方法> <hash key> <enc key> <加密之前的file> <加密之后的file>
    如：
    ./rc4_encryptor encrypt_rc4 1234532 343456 resnet50.mge resnet50_encrypted.mge

rc4_encryptor 加密工具中可以使用下面 4 中方式加密：

* encrypt_predefined_rc4 ：使用内部预定义 RC4 加密方式和预定义的 hash key 和 enc key，不需要传入秘钥。
* encrypt_predefined_sfrc4：使用内部预定义 simple_fast_RC4 加密方式和内部预定义的 hash key 和 enc key，不需要传入秘钥。
* encrypt_rc4 ：使用内部预定义 RC4 加密方式，需要用户传入两个key。
* encrypt_sfrc4：加密方式使用内部定义的 simple_fast_RC4 加密方法，需要用户传入两个key。
  
其中 hash key 和 enc key 均为 uint64_t 类型数字。其中 encrypt_predefined_rc4 和 encrypt_rc4 使用的是 RC4 加密方式，
其中 encrypt_predefined_sfrc4 和 encrypt_sfrc4 使用的是 simple_fast_RC4 加密方式，只是他们的秘钥来源不一样。

其他加密方式
^^^^^^^^^^^^^^^^^^

MegEngine Lite 支持用户自定义的加密方式，但是需要在 Network 载入模型之前将解密方式和解密秘钥注册到 MegEngine Lite 中。

解密并载入模型
---------------------

通过MegEngine Lite 解密上面加密之后的模型，只需要在 :ref:`basic_code` 的基础上增加简单的配置就可以完成。

解密预定义的加密方式和秘钥加密模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该方式加密的模型只需要在 Network 创建时候的 config 中指明具体解密算法的名字。

.. code-block:: cpp

    Config config;
    config.bare_model_cryption_name = "AES_default";

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);

.. code-block:: python

    from megenginelite import *

    config = LiteConfig()
    config.bare_model_cryption_name = "AES_default".encode("utf-8")
    network = LiteNetwork(config)
    network.load(model_path)

在 Network 的 load_model 中将对模型进行解密并载入模型，上面是解密通过 AES 并使用默认的 AES 秘钥进行加密的模型并载入。
主要是配置模型载入时候的 config 中的 bare_model_cryption_name 成员，
MegEngine Lite 中支持的 bare_model_cryption_name 可以是："AES_default"，"RC4_default"，"SIMPLE_FAST_RC4_default"。

解密预定义的加密方式加密和自定义秘钥加密的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该方式加密的模型只需要在 Network 创建时候的 config 中指明具体解密算法的名字，以及更新对应解密算法使用到的秘钥。

.. code-block:: cpp

    uint64_t hash_key = xxx;
    uint64_t enc_key = xxxx;
    std::vector<int8_t> key(16, 0);
    uint64_t* ptr = static_cast<uint64_t*>(key.data());
    ptr[0] = hash_key;
    ptr[1] = enc_key;
    update_decryption_or_key("RC4_default", nullptr, key);

    Config config;
    config.bare_model_cryption_name = "RC4_default";
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);

.. code-block:: python

    from megenginelite import *

    new_key = [1]*16
    LiteGlobal.update_decryption_key("RC4_default", new_key)

    config = LiteConfig()
    config.bare_model_cryption_name = "RC4_default".encode("utf-8")
    network = LiteNetwork(config)
    network.load(model_path)

更新 MegEngine Lite 里面的解密算法的秘钥主要使用 update_decryption_or_key 接口，参考：:ref:`update_decryption_or_key` ;


解密自定义的加密方式加密的模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用户如果使用自定义的加密算法加密的模型，则用户需要将对应的解密算法和秘钥注册到 MegEngine Lite 中，才能进行解密。

.. code-block:: cpp

    std::vector<uint8_t> decrypt_model(
            const void* model_mem, size_t size, const std::vector<uint8_t>& key) {
        if (key.size() == 1) {
            std::vector<uint8_t> ret(size, 0);
            const uint8_t* ptr = static_cast<const uint8_t*>(model_mem);
            uint8_t key_data = key[0];
            for (size_t i = 0; i < size; i++) {
                ret[i] = ptr[i] ^ key_data ^ key_data;
            }
            return ret;
        } else {
            printf("the user define decrypt method key length is wrong.\n");
            return {};
        }
    }
    register_decryption_and_key("just_for_test", decrypt_model, {15});

    Config config;
    config.bare_model_cryption_name = "just_for_test";
    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    network->load_model(model_path);

.. code-block:: python

    from megenginelite import *

    @decryption_func
    def decrypt_model(in_arr, key_arr, out_arr):
        if not out_arr:
            return in_arr.size
        else:
            for i in range(in_arr.size):
                out_arr[i] = in_arr[i] ^ key_arr[0] ^ key_arr[0]
            return out_arr.size

    LiteGlobal.register_decryption_and_key("just_for_test", decrypt_model, [15])
    config = LiteConfig()
    config.bare_model_cryption_name = "just_for_test".encode("utf-8")

    network = LiteNetwork()
    model_path = os.path.join(self.source_dir, "shufflenet.mge")
    network.load(model_path)

上面分别展示在 CPP 和 python 中分别是用一个假的解密方法 decrypt_model 来展示使用自动义解密方式的过程。

模型打包
---------------

上面介绍了对单个模型进行加密和解密的功能，下面在模型加解密的基础上，将模型的基本信息，配置信息，IO 信息，以及用户自定义的信息和
模型一同打包在一起，MegEngine Lite 在载入这些模型时候将设置 Network 的这些信息，不需要用户再手动设置这些信息，实现模型自解释功能。
其中这些模型信息的解密方式可以用户自定义，也可以使用 MegEngine Lite 默认定义的方法进行解析，使用 MegEngine Lite 默认的解析
方法，用户也可以添加自己定义的额外信息，并通过 :ref:`get_model_extra_info` 接口可以获取到并自行解析。

模型结构
^^^^^^^^^^^^^^^

打包之后的模型将会是用 `flatbuffer <https://github.com/google/flatbuffers>`_ 进行序列化，在载入时也将使用 flatbuffer 进行反序列化，下面
是 MegEngine Lite 定义的模型序列化格式 :src:`lite/src/parse_model/pack_model.fbs`。

由 :src:`lite/src/parse_model/pack_model.fbs` 可知，打包之后的模型主要由 ：

* ModelHeader：打包时候的一些信息，包括：模型的名字，模型信息的解密方式名字，模型信息的解析方法名字，模型的解密方法名字
* ModelInfo：打包时候指定的模型信息文件的数据，这个文件也可以选择加密。
* ModelData：打包时候指定的模型的数据，模型文件可以选择加密和不加密。

ModelInfo
^^^^^^^^^^^^^^^^^^^^

ModelInfo 的格式可以用户自定义，如果用户自定义 ModelInfo 的格式，那么用户就必须通过 :ref:`register_parse_info_func` 注册解析模型信息的函数
到 MegEngine Lite 中。目前 MegEngine Lite 中也预先定义好了一个解析模型信息的方法，这个方法名字为："LITE_default"。下面是 LITE_default 支持
模型信息的格式，是一个 `JSON <https://www.json.org/json-en.html>`_ 文件，部分信息是必须的，部分是可选择的。

.. code-block::

    {
        "name": "shufflenet_test",
        "valid": true,
        "version": "8.9999.0",
        "has_compression": false,
        "device": {
            "type": "CPU",
            "device_id": 0,
            "number_threads": 1,
            "use_tensor_rt": false,
            "enable_inplace_model": false
        },
        "options":{
            "weight_preprocess": false,
            "var_sanity_check_first_run": true,
            "const_shape": false,
            "jit_level": 0,
            "record_level": 0
        },
        "IO":{
            "inputs":[
                {
                    "name": "data",
                    "io_type": "value",
                    "is_host": true,
                    "dtype": "float32",
                    "shape": {
                        "dim0": 1,
                        "dim1": 3,
                        "dim2": 224,
                        "dim3": 224
                    }
                }
            ],
            "outputs":[
                {
                    "name": "TRUE_DIV(EXP[12065],reduce0[12067])[12077]",
                    "io_type": "value",
                    "is_host": true,
                    "dtype": "float32",
                    "shape": {
                        "dim0": 1,
                        "dim1": 1000,
                        "dim2": 0,
                        "dim3": 0
                    }
                }
            ]
        },
        "extra_info":{
            ....用户自定义的部分。
        }
    }

这里面大多数都是可选的，只有：name，valid，version 是必须的，其他部分都是可选的。这些配置主要对应 Network 中的：

* 模型运行设备：device 信息。
* 模型优化选项：options 配置。
* 模型 IO 配置：模型中输入输出 Tensor 信息配置。
* 额外的信息：用户自定义的额外信息，可以通过调用 :ref:`get_model_extra_info` 接口可以获取到并自行解析。

打包
^^^^^^^

MegEngine Lite 中 :src:`lite/tools/pack_model/pack_model_and_info.py` 脚本可以支持快速完成模型打包，可以直接用其对已有的模型和模型 Info 的文件进行打包，
用户需要指定：

* 模型名字：如果有模型 info，则需要和 model info 中名字匹配，否则会 check 失败。
* 模型加密方式 ：Lite 中目前包含的加密方式名字为：ES_default，RC4_default，SIMPLE_FAST_RC4_default，如果是自定义加密方式，则写对应的加密方式名字。
* 模型info文件加密方式：Lite中目前包含的加密方式名字为：ES_default，RC4_default，SIMPLE_FAST_RC4_default。
* 模型info 解析方式信息：Lite中目前只有：LITE_default。

.. warning::

    * 使用 :src:`lite/tools/pack_model/pack_model_and_info.py` 工具，需要用户编译安装flatbuffers，编译安装很简单，详见 `教程 <https://google.github.io/flatbuffers/flatbuffers_guide_building.html>_`，
    * 并将编译之后的可执行文件 flatc 的路径加载到系统的 $PATH 中。
    * 并且安装 Python 版本 flatbuffers。 python3 -m pip install flatbuffers。

.. code-block:: bash

    python3 -m pip3 install flatbuffers
    python3 pack_model_and_info.py --input-model xxx.mge \
        --model-name="shufflenet_test" \
        --model-cryption="RC4_default" \
        --input-info xxx.json \
        --info-cryption="RC4_default" \
        --info-parser="LITE_default" \
        -o packed.lite

上面的打包时候指定了：模型的名字为：shufflenet_test， 模型的解密方式为：RC4_default，模型的Info为：xxx.json， 
Info的解密方式为：RC4_default，info的解析方式为：LITE_defaul，最后输出 packed.lite 为打包之后的模型文件。

.. note::

    * 如果模型没有加密则可以不用指定模型加密方式。
    * 如果模型没有模型信息文件，则可以不用指定 --input-info，--info-cryption，--info-parser。

打包之后的模型加载
^^^^^^^^^^^^^^^^^^^^^^^^^^

打包之后的模型文件的加载和正常模型的加载完全一样，
参考 :ref:`lite-quick-start-cpp`，:ref:`lite-quick-start-python`，并可以省掉配置这种模型信息的过程。
