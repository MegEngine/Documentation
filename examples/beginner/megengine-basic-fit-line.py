import numpy as np
import megengine.functional as F
from megengine import Tensor, Parameter
from megengine.autodiff import GradManager
import megengine.optimizer as optim


np.random.seed(20200325)


def get_point_examples(w=5.0, b=2.0, nums_eample=100, noise=5):

    x = np.zeros((nums_eample,))
    y = np.zeros((nums_eample,))

    for i in range(nums_eample):
        x[i] = np.random.uniform(-10, 10)
        y[i] = w * x[i] + b + np.random.uniform(-noise, noise)

    return x, y


x, y = get_point_examples()

w = Parameter(0.0)
b = Parameter(0.0)


def f(x):
    return w * x + b


gm = GradManager().attach([w, b])
optimizer = optim.SGD([w, b], lr=0.01)

nums_epoch = 5
for epoch in range(nums_epoch):
    x = Tensor(x)
    y = Tensor(y)

    with gm:
        pred = f(x)
        loss = F.nn.square_loss(pred, y)
        gm.backward(loss)
        optimizer.step().clear_grad()

    print(f"Epoch = {epoch}, \
            w = {w.item():.3f}, \
            b = {b.item():.3f}, \
            loss = {loss.item():.3f}")
