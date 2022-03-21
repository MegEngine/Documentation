#include <jni.h>
#include <string>
#include "lite/network.h"
#include "lite/tensor.h"
#include "ImageNetLabels.h"

using namespace lite;

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_example_cameraxapp_ImageClassifier_predict(
        JNIEnv *env,
        jobject thiz,
        jbyteArray model,
        jintArray image,
        jint height,
        jint width) {

    jboolean isCopy = JNI_FALSE;
    jbyte *const model_data = env->GetByteArrayElements(model, &isCopy);
    jint *const image_data = env->GetIntArrayElements(image, &isCopy);

    std::shared_ptr<Network> network = std::make_shared<Network>();
    network->load_model(model_data, env->GetArrayLength(model));

    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor("data");
    Layout input_layout = input_tensor->get_layout();
    input_layout.shapes[0] = height;
    input_layout.shapes[1] = width;
    input_tensor->set_layout(input_layout);

    size_t length = input_tensor->get_tensor_total_size_in_byte();
    auto in_data_ptr = static_cast<uint8_t *>(input_tensor->get_memory_ptr());
    for (size_t i = 0; i < length; i++) {
        in_data_ptr[i] = (uint8_t)(image_data[i]);
    }

    network->forward();
    network->wait();

    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    auto* predict_ptr = static_cast<float*>(output_tensor->get_memory_ptr());

    ImageNetLabels label;
    float max_prob = 0;
    std::string max_class;

    const int numClasses = 1000;
    for (size_t i = 0; i < numClasses; i++) {
        if (predict_ptr[i] > max_prob) {
            max_prob = predict_ptr[i];
            max_class = label.imagenet_labelstring(i);
        }
    }

    env->ReleaseByteArrayElements(model, model_data, JNI_ABORT);
    env->ReleaseIntArrayElements(image, image_data, JNI_ABORT);
    return env->NewStringUTF(max_class.c_str());
}

}