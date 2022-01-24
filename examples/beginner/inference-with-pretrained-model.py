import cv2
import numpy as np

import megengine
import megengine.data.transform as T
import megengine.functional as F
import megengine.hub as hub


image_path = "/path/to/example.jpg"  # Select the same image to read
image = cv2.imread(image_path)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[103.530, 116.280, 123.675],
                std=[57.375, 57.120, 58.395]),
    T.ToMode("CHW"),
])

model = hub.load('megengine/models', 'resnet18', pretrained=True)
model.eval()

processed_img = transform.apply(image)  # Still NumPy ndarray here
processed_img = F.expand_dims(megengine.Tensor(processed_img), 0)  # -> 1CHW
logits = model(processed_img)
probs = F.softmax(logits)
print(probs)
