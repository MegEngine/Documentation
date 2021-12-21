import megengine
import megengine.data as data
import megengine.data.transform as T
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import megengine.autodiff as autodiff

from megengine.data.dataset import MNIST

MNIST_DATA_PATH = "/data/datasets/MNIST"

train_dataset = MNIST(MNIST_DATA_PATH, train=True, download=False)
test_dataset = MNIST(MNIST_DATA_PATH, train=False, download=False)

train_sampler = data.RandomSampler(train_dataset, batch_size=64)
test_sampler = data.SequentialSampler(test_dataset, batch_size=4)

transform = T.Compose([
    T.Normalize(0.1307*255, 0.3081*255),
    T.Pad(2),
    T.ToMode("CHW"),
])

train_dataloader = data.DataLoader(train_dataset, train_sampler, transform)
test_dataloader = data.DataLoader(test_dataset, test_sampler, transform)


class LeNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Conv2d(1, 6, 5)
        self.conv2 = M.Conv2d(6, 16, 5)
        self.fc1 = M.Linear(16 * 5 * 5, 120)
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


model = LeNet()
gm = autodiff.GradManager().attach(model.parameters())
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)

epochs = 10
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_data, batch_label in train_dataloader:
        batch_data = megengine.Tensor(batch_data)
        batch_label = megengine.Tensor(batch_label)

        with gm:
            logits = model(batch_data)
            loss = F.nn.cross_entropy(logits, batch_label)
            gm.backward(loss)
            optimizer.step().clear_grad()

        total_loss += loss.item()

    print(f"Epoch: {epoch}, loss: {total_loss/len(train_dataset)}")


model.eval()
correct, total = 0, 0
for batch_data, batch_label in test_dataloader:
    batch_data = megengine.Tensor(batch_data)
    batch_label = megengine.Tensor(batch_label)

    logits = model(batch_data)
    pred = F.argmax(logits, axis=1)
    correct += (pred == batch_label).sum().item()
    total += len(pred)

print(f"Correct: {correct}, total: {total}, accuracy: {float(correct)/total}")
