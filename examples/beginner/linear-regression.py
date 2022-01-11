from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import megengine
import megengine.functional as F
import megengine.data.transform as T
import megengine.optimizer as optim
import megengine.autodiff as autodiff

from megengine.data import DataLoader
from megengine.data.dataset import ArrayDataset
from megengine.data.sampler import SequentialSampler


# Get thr original dataset
DATA_PATH = "/data/datasets/california/"
X, y = fetch_california_housing(data_home=DATA_PATH, return_X_y=True)

# Split dataset to train/val/test dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=37)
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  random_state=37)
nums_feature = X_train.shape[1]

# Prepare dataloader
transform = T.Normalize(mean=X_train.mean(), std=X_train.std())

train_dataset = ArrayDataset(X_train, y_train)
train_sampler = SequentialSampler(train_dataset, batch_size=128)
train_dataloader = DataLoader(train_dataset, train_sampler, transform)

val_dataset = ArrayDataset(X_val, y_val)
val_sampler = SequentialSampler(val_dataset, batch_size=128)
val_dataloader = DataLoader(val_dataset, val_sampler, transform)

test_dataset = ArrayDataset(X_test, y_test)
test_sampler = SequentialSampler(test_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, test_sampler, transform)

# Parameter initialization
w = megengine.Parameter(F.zeros((nums_feature,)))
b = megengine.Parameter(0.0)


# Define linear regression model
def linear_model(X):
    return F.matmul(X, w) + b


# GradManager and Optimizer setting
gm = autodiff.GradManager().attach([w, b])
optimizer = optim.SGD([w, b], lr=0.01)

# Training
nums_epoch = 10
for epoch in range(nums_epoch):
    training_loss = 0
    validation_loss = 0

    for step, (X, y) in enumerate(train_dataloader):
        X = megengine.Tensor(X)
        y = megengine.Tensor(y)

        with gm:
            pred = linear_model(X)
            loss = F.nn.square_loss(pred, y)
            gm.backward(loss)
            optimizer.step().clear_grad()

        training_loss += loss.item() * len(X)

        if step % 30 == 0:
            print(f"Epoch = {epoch}, step = {step}, loss = {loss.item()}")

    for X, y in val_dataloader:
        X = megengine.Tensor(X)
        y = megengine.Tensor(y)

        pred = linear_model(X)
        loss = F.nn.l1_loss(y, pred)

        validation_loss += loss.item() * len(X)

    training_loss /= len(X_train)
    validation_loss /= len(X_val)

    print(f"Epoch = {epoch},"
          f"training_loss = {training_loss},"
          f"validation_loss = {validation_loss}")


# Test
test_loss = 0
for X, y in test_dataloader:
    X = megengine.Tensor(X)
    y = megengine.Tensor(y)

    pred = linear_model(X)
    loss = F.nn.l1_loss(y, pred)
    test_loss += loss.item() * len(X)

test_loss /= len(X_test)

print(f"Test_loss = {test_loss}")
