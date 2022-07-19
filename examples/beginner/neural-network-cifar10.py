import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import megengine.autodiff as autodiff

from os.path import expanduser
DATA_PATH = expanduser("~/data/datasets/CIFAR10")

train_dataset = data.dataset.CIFAR10(DATA_PATH, train=True)
test_dataset = data.dataset.CIFAR10(DATA_PATH, train=False)

train_sampler = data.RandomSampler(train_dataset, batch_size=64)
test_sampler = data.SequentialSampler(test_dataset, batch_size=64)

"""
import nump as np

X_train, y_train = map(np.array, train_dataset[:])
mean = [X_train[:,:,:,i].mean() for i in range(3)]
std = [X_train[:,:,:,i].std() for i in range(3)]
"""

transform = T.Normalize([113.86538318359375, 122.950394140625, 125.306918046875],
                        [66.70489964063091, 62.08870764001421, 62.993219278136884])

train_dataloader = data.DataLoader(train_dataset, train_sampler, transform)
test_dataloader = data.DataLoader(test_dataset, test_sampler, transform)

num_features = train_dataset[0][0].size
num_hidden = 256
num_classes = 10


# Define model
class NN(M.Module):
    def __init__(self):
        super().__init__()
        self.fc = M.Linear(num_features, num_hidden)
        self.classifier = M.Linear(num_hidden, num_classes)

    def forward(self, x):
        x = F.nn.relu(self.fc(x))
        x = self.classifier(x)
        return x


model = NN()

# GradManager and Optimizer setting
gm = autodiff.GradManager().attach(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training and validation
nums_epoch = 50
for epoch in range(nums_epoch):
    training_loss = 0
    nums_train_correct, nums_train_example = 0, 0
    nums_val_correct, nums_val_example = 0, 0

    for step, (image, label) in enumerate(train_dataloader):
        image = F.flatten(megengine.Tensor(image), 1)
        label = megengine.Tensor(label)

        with gm:
            score = model(image)
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
        pred = F.argmax(model(image), axis=1)

        nums_val_correct += (pred == label).sum().item()
        nums_val_example += len(image)

    val_acc = nums_val_correct / nums_val_example

    print(f"Epoch = {epoch}, "
          f"train_loss = {training_loss:.3f}, "
          f"train_acc = {training_acc:.3f}, "
          f"val_acc = {val_acc:.3f}")
