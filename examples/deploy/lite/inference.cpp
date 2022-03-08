#include <cstdlib>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2//opencv.hpp>
#include "lite/network.h"
#include "lite/tensor.h"
#include "ImageNetLabels.h"

using namespace lite;

int main (int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: ./inference <model_path> <image_path>" << std::endl;
    return -1;
  }

  std::string model_path = argv[1];

  //! Create and load the networ3
  std::shared_ptr<Network> network = std::make_shared<Network>();

  //! Load the model
  network->load_model(model_path);

  //! Get the input tensor of the network with name "data"
  std::shared_ptr<Tensor> input_tesnor = network->get_io_tensor("data");

  //! Read image data, preprocess and fill to input tensor
  cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);

  //! 1. Resize --
  const int resizeWidth = 256, resizeHeight = 256;
  cv::Size scale(256, 256);
  cv::resize(image, image, scale, 0, 0, cv::INTER_LINEAR);

  //! 2. Center crop
  const int cropSize = 224;
  const int offsetW = (image.cols - cropSize) / 2.0;
  const int offsetH = (image.rows - cropSize) / 2.0;
  const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
  image = image(roi).clone();
  image.convertTo(image, CV_32FC3);  // Usigned int to float
  
  //! Fill image data to input tensor
  float* imgptr = image.ptr<float>();
  float* in_data_ptr = static_cast<float*>(input_tesnor->get_memory_ptr());

  //! 3 & 4. Normalize and ToMode("HWC" to "CHW")
  const float mean[] = {103.530f, 116.280f, 123.675f}; // BGR
  const float std[] = {57.375f, 57.120f, 58.395f};
  size_t pixels = cropSize * cropSize;
  for (size_t i = 0; i < pixels; i++) {
    for (size_t j = 0; j < 3; j++) {
      in_data_ptr[i + pixels * j] = (imgptr[3 * i + j] - mean[j]) / std[j];
    }
  }
  
  //! Forward (inference)
  network->forward();
  network->wait();

  //! Get the inference output tensor of index 0
  std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
  float* predict_ptr = static_cast<float*>(output_tensor->get_memory_ptr());

  //! Output
  ImageNetLabels label;
  float max_prob = predict_ptr[0];
  std::string max_class = label.imagenet_labelstring(0);

  const int numClasses = 1000;
  const float threshold = 0.02;
  for (size_t i = 0; i < numClasses; i++) {
    float cur_prob = predict_ptr[i];
    std::string cur_label = label.imagenet_labelstring(i);

    if (cur_prob > max_prob) {
      max_prob = cur_prob;
      max_class = cur_label;
    }
    if (cur_prob > threshold) {
      std::cout << "The class " << cur_label
        << "with probability = "<< 100 * cur_prob << "%" << std::endl;
    }
  }

  std::cout << "The final predicted class is " << max_class 
    << " with probability = " << 100 * max_prob << "%" << std::endl;

  return 0;
}
