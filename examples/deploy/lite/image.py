import sys

import cv2
import numpy as np

import megengine.data.transform as T


def main():

    path = sys.path[0] + "/../../../source/_static/images/cat.jpg"

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
    input_data = input_data.flatten()

    print(input_data[:5])


if __name__ == "__main__":
    main()
