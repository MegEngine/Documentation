import sys
import argparse

import cv2
import numpy as np

import megengine as mge
import megengine.functional as F

from model import LeNet

logging = mge.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Infer an input handwritten image with trained model.")
    parser.add_argument("-m", "--model", default=None, type=str)
    parser.add_argument("-i", "--image", default=None, type=str)
    args = parser.parse_args()

    model = LeNet()
    if args.model is not None:
        logging.info("load from checkpoint %s", args.model)
        checkpoint = mge.load(args.model)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

    if args.image is None:
        path = sys.path[0] + \
            "/../../../source/_static/images/handwrittern-digit.png"
        logging.info("Input image was not given, use the default example.")
    else:
        path = args.image

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    processed_img = process(image)
    processed_img = mge.Tensor(processed_img).reshape((1, 1, 32, 32))

    def infer_func(processed_img):
        logit = model(processed_img)
        label = F.argmax(logit).item()
        return label

    label = infer_func(processed_img)

    print(f"The predicted class is {label}.")


def process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    image = np.array(255 - image)
    return image


if __name__ == "__main__":
    main()
