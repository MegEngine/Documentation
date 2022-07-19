import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
import megengine.optimizer as optim
import megengine.autodiff as autodiff


# Get train and test dataset and prepare dataloader
from os.path import expanduser
DATA_PATH = expanduser("~/data/datasets/MNIST")

train_dataset = data.dataset.MNIST(DATA_PATH, train=True)
test_dataset = data.dataset.MNIST(DATA_PATH, train=False)

train_sampler = data.RandomSampler(train_dataset, batch_size=64)
test_sampler = data.SequentialSampler(test_dataset, batch_size=64)

transform = T.Normalize(33.318421449829934, 78.56748998339798)

train_dataloader = data.DataLoader(train_dataset, train_sampler, transform)
test_dataloader = data.DataLoader(test_dataset, test_sampler, transform)

num_features = train_dataset[0][0].size
num_classes = 10


# Parameter initialization
W = megengine.Parameter(F.zeros((num_features, num_classes)))
b = megengine.Parameter(F.zeros((num_classes,)))


# Define linear classification model
def linear_cls(X):
    return F.matmul(X, W) + b


# GradManager and Optimizer setting
gm = autodiff.GradManager().attach([W, b])
optimizer = optim.SGD([W, b], lr=0.01)


# Training and validation
nums_epoch = 5
for epoch in range(nums_epoch):
    training_loss = 0
    nums_train_correct, nums_train_example = 0, 0
    nums_val_correct, nums_val_example = 0, 0

    for step, (image, label) in enumerate(train_dataloader):
        image = F.flatten(megengine.Tensor(image), 1)
        label = megengine.Tensor(label)

        with gm:
            score = linear_cls(image)
            loss = F.nn.cross_entropy(score, label)
            gm.backward(loss)
            optimizer.step().clear_grad()

        training_loss += loss.item() * len(image)

        pred = F.argmax(score, axis=1)
        nums_train_correct += (pred == label).sum().item()
        nums_train_example += len(image)

    training_acc = nums_train_correct / nums_train_example
    training_loss /= nums_train_example

    for image, label in test_dataloader:
        image = F.flatten(megengine.Tensor(image), 1)
        label = megengine.Tensor(label)
        pred = F.argmax(linear_cls(image), axis=1)

        nums_val_correct += (pred == label).sum().item()
        nums_val_example += len(image)

    val_acc = nums_val_correct / nums_val_example

    print(f"Epoch = {epoch}, "
          f"train_loss = {training_loss:.3f}, "
          f"train_acc = {training_acc:.3f}, "
          f"val_acc = {val_acc:.3f}")
