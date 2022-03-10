import sys
import argparse
import json

import numpy as np
import cv2

import megengine.functional as F
import megengine.data.transform as T
import megengine.hub as hub

from megengine import Tensor
from megengine.jit import trace


def main():

    parser = argparse.ArgumentParser(
        description="Get pretrained model from hub and dump .mge file")
    parser.add_argument(
        "--repo",
        "--repository",
        metavar="REPO",
        type=str,
        default="megvii-research/basecls",
        help="The repository provide hubconf.py file."
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        type=str,
        default="snetv2_x100",
        help="The pretrained model to be used."
    )
    parser.add_argument(
        "--image",
        metavar="IMAGE",
        type=str,
        help="The input data used for tracing"
    )

    args = parser.parse_args()

    net = hub.load(args.repo, args.model, pretrained=True)
    net.eval()

    @trace(symbolic=True, capture_as_const=True)
    def infer_func(data, *, model):
        pred = model(data)
        pred_normalized = F.softmax(pred)
        return pred_normalized

    if args.image is not None:
        path = args.image
    else:
        # data = np.random.random([1, 3, 224, 224]).astype(np.float32)
        path = sys.path[0] + "/../../../source/_static/images/cat.jpg"

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[103.530, 116.280, 123.675],
                    std=[57.375, 57.120, 58.395]),  # BGR
        T.ToMode("CHW"),
    ])
    data = transform.apply(image)[np.newaxis, :]

    probs = infer_func(Tensor(data), model=net)
    infer_func.dump(args.model + "_deploy.mge", arg_names=["data"])

    top_probs, classes = F.topk(probs, k=5, descending=True)
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
