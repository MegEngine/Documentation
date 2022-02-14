import megengine.data as data
import megengine.data.transform as T

from megengine.data.dataset import MNIST


def build_dataset(path, batch_size=64):
    """Build train and test dataloader from MNIST dataset."""

    transform = T.Compose([
        T.Normalize(0.1307*255, 0.3081*255),
        T.Pad(2),
        T.ToMode("CHW"),
    ])

    train_dataset = MNIST(path, train=True)
    train_sampler = data.RandomSampler(
        train_dataset, batch_size=batch_size)
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=transform,
    )

    test_dataset = MNIST(path, train=False)
    test_sampler = data.SequentialSampler(
        test_dataset, batch_size=batch_size)
    test_dataloader = data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        transform=transform,
    )

    return train_dataloader, test_dataloader
