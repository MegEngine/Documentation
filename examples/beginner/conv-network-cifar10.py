import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import megengine.autodiff as autodiff

DATA_PATH = "/data/datasets/CIFAR10/"

train_dataset = data.dataset.CIFAR10(root=DATA_PATH, train=True)
test_dataset = data.dataset.CIFAR10(root=DATA_PATH, train=False)

train_sampler = data.RandomSampler(train_dataset, batch_size=64)
test_sampler = data.SequentialSampler(test_dataset, batch_size=64)

"""
import nump as np

X_train, y_train = map(np.array, train_dataset[:])
mean = [X_train[:,:,:,i].mean() for i in range(3)]
std = [X_train[:,:,:,i].std() for i in range(3)]
"""

transform = T.Compose([
    T.Normalize([113.86538318359375, 122.950394140625, 125.306918046875],
                [66.70489964063091, 62.08870764001421, 62.993219278136884]),
    T.ToMode("CHW"),
])

train_dataloader = data.DataLoader(train_dataset, train_sampler, transform)
test_dataloader = data.DataLoader(test_dataset, test_sampler, transform)


# Define model
class ConvNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Conv2d(3, 6, 5)
        self.conv2 = M.Conv2d(6, 16, 5)
        self.fc1 = M.Linear(16*5*5, 120)
        self.fc2 = M.Linear(120, 84)
        self.classifier = M.Linear(84, 10)

        self.relu = M.ReLU()
        self.pool = M.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = F.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        return x


model = ConvNet()

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
        image = megengine.Tensor(image)
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
        image = megengine.Tensor(image)
        label = megengine.Tensor(label)
        pred = F.argmax(model(image), axis=1)

        nums_val_correct += (pred == label).sum().item()
        nums_val_example += len(image)

    val_acc = nums_val_correct / nums_val_example

    print(f"Epoch = {epoch}, "
          f"train_loss = {training_loss:.3f}, "
          f"train_acc = {training_acc:.3f}, "
          f"val_acc = {val_acc:.3f}")
