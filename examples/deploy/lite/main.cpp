#include <cstdlib>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include "lite/network.h"
#include "lite/tensor.h"

using namespace lite;

int main (int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: ./lite_softmax <model_path>" << std::endl;
    return -1;
  }

  std::string model_path = argv[1];

  //! Create and load the network
  std::shared_ptr<Network> network = std::make_shared<Network>();

  //! Load the model
  network->load_model(model_path);

  //! Get the input tensor of the network with name "data"
  std::shared_ptr<Tensor> input_tesnor = network->get_io_tensor("data");

  //! Fill random data to input tensor
  srand(static_cast<unsigned>(time(NULL)));
  size_t length = input_tesnor->get_tensor_total_size_in_byte() / sizeof(float);
  float* in_data_ptr = static_cast<float*>(input_tesnor->get_memory_ptr());
  for (size_t i = 0; i < length; i++) {
    in_data_ptr[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
  }

  //! Forward (inference)
  network->forward();
  network->wait();

  //! Get the inference output tensor of index 0
  std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
  float* predict_ptr = static_cast<float*>(output_tensor->get_memory_ptr());

  float sum = 0.0f, max = predict_ptr[0];
  for (size_t i = 0; i < 1000; i++) {
    sum += predict_ptr[i];
    if (predict_ptr[i] > max) {
      max = predict_ptr[i];
    }
  }

  std::cout << "The output SUM is " << sum << ", MAX is " << max << std::endl;

  return 0;
}
