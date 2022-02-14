import argparse

import megengine as mge
import megengine.functional as F

from data import build_dataset
from model import LeNet

logging = mge.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Test MegEngine LeNet Model on MNIST dataset")
    parser.add_argument("-d", "--data", help="path to MNIST dataset")
    parser.add_argument(
        "-m", "--model",
        metavar="PKL",
        default=None,
        help="path to model checkpoint"
    )

    args = parser.parse_args()

    # Build dataset
    _, test_dataloader = build_dataset(args.data)

    # Build model and load from checkpoint
    model = LeNet()
    if args.model is not None:
        logging.info("load from checkpoint %s", args.model)
        checkpoint = mge.load(args.model)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

    # Valid function
    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        pred = F.argmax(logits, axis=1)
        correct = (pred == label).sum().item()
        return loss, correct

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
    logging.info("Valid loss: %.6f valid acc %.6f",
                 valid_loss,
                 valid_acc,
                 )


if __name__ == "__main__":
    main()
