import sys
import argparse
import json

import cv2
import numpy as np

import megengine.functional as F
import megengine.data.transform as T
from megengine import Tensor
from megenginelite import LiteNetwork

import logging


def main():

    parser = argparse.ArgumentParser(
        description="Test the lite python interface")
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-i", "--image", default=None, type=str)
    args = parser.parse_args()

    network = LiteNetwork()
    network.load(args.model)

    if args.image is None:
        path = sys.path[0] + \
            "/../../../source/_static/images/cat.jpg"
        logging.info("Input image was not given, use the default example.")
    else:
        path = args.image

    # Get input data and transform to Lite Tensor
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[103.530, 116.280, 123.675],
                    std=[57.375, 57.120, 58.395]),  # BGR
        T.ToMode("CHW"),
    ])
    input_data = transform.apply(image)[np.newaxis, :]  # CHW -> 1CHW
    input_tensor = network.get_io_tensor("data")
    input_tensor.set_data_by_copy(input_data)

    # Inference
    network.forward()
    network.wait()

    output_names = network.get_all_output_name()
    output_tensor = network.get_io_tensor(output_names[0])
    output_data = output_tensor.to_numpy()
    output_tensor = Tensor(output_data)  # probs

    # Show predicted result (top5-classes)
    top_probs, classes = F.topk(output_tensor, k=5, descending=True)
    with open("../assets/imagenet_class_info.json") as fp:
        imagenet_class_index = json.load(fp)

    for rank, (prob, classid) in enumerate(
        zip(top_probs.numpy().reshape(-1), classes.numpy().reshape(-1))
    ):
        print("{}: class = {:20s} with probability = {:4.7f} %".format(
            rank, imagenet_class_index[str(classid)][1], 100 * prob
        ))


if __name__ == "__main__":
    main()
