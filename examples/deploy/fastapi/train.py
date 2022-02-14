import argparse
import os

import megengine as mge
import megengine.autodiff as autodiff
import megengine.optimizer as optim
import megengine.functional as F

from data import build_dataset
from model import LeNet

logging = mge.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Train MegEngine LeNet Model on MNIST dataset")
    parser.add_argument("-d", "--data", help="path to MNIST dataset")
    parser.add_argument(
        "--save",
        metavar="DIR",
        default="output",
        help="path to save checkpoint and log",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs to run (default: 90)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=64,
        type=int,
        help="batch size for single GPU (default: 64)")

    parser.add_argument(
        "--lr",
        "--learning-rate",
        metavar="LR",
        default=0.025,
        type=float,
        help="learning rate for single GPU (default: 0.025)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-4,
        type=float,
        help="weight decay (default: 1e-4)"
    )

    args = parser.parse_args()

    # Logging config
    os.makedirs(args.save, exist_ok=True)
    mge.logger.set_log_file(os.path.join(args.save, "log.txt"))

    # Build dataset
    train_dataloader, test_dataloader = build_dataset(
        args.data, args.batch_size)

    # Build model
    model = LeNet()

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
    )

    # Optimizer
    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Train and valid function
    def train_step(image, label):
        with gm:
            logits = model(image)
            loss = F.nn.cross_entropy(logits, label)
            gm.backward(loss)
            opt.step().clear_grad()
        pred = F.argmax(logits, axis=1)
        correct = (pred == label).sum().item()
        return loss, correct

    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        pred = F.argmax(logits, axis=1)
        correct = (pred == label).sum().item()
        return loss, correct

    # Start training and validation in each epoch
    for epoch in range(args.epochs):

        model.train()
        total_loss, total_num, correct_num = 0, 0, 0
        for image, label in train_dataloader:
            image = mge.Tensor(image)
            label = mge.Tensor(label)
            loss, correct = train_step(image, label)
            total_loss += loss.item()
            correct_num += correct
            total_num += len(label)
        train_loss = float(total_loss)/total_num
        train_acc = float(correct_num)/total_num

        model.eval()
        total_loss, total_num, correct_num = 0, 0, 0
        for image, label in test_dataloader:
            image = mge.Tensor(image)
            label = mge.Tensor(label)
            loss, correct = valid_step(image, label)
            total_loss += loss.item()
            correct_num += correct
            total_num += len(label)
        valid_loss = float(total_loss)/total_num
        valid_acc = float(correct_num)/total_num

        # Save logging information
        logging.info("Epoch: %02d train loss: %.6f train acc: %.6f valid loss: %.6f valid acc %.6f",
                     epoch,
                     train_loss,
                     train_acc,
                     valid_loss,
                     valid_acc,
                     )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            mge.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
            },
                os.path.join(args.save, "checkpoint.pkl"))


if __name__ == "__main__":
    main()
