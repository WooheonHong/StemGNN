import os
import torch
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from datetime import datetime
from models.handler import tuning, test
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--evaluate", type=bool, default=True)
parser.add_argument("--dataset", type=str, default="ECG_data")
parser.add_argument("--window_size", type=int, default=12)
parser.add_argument("--horizon", type=int, default=3)
parser.add_argument("--train_length", type=float, default=7)
parser.add_argument("--valid_length", type=float, default=2)
parser.add_argument("--test_length", type=float, default=1)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--multi_layer", type=int, default=5)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--validate_freq", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--norm_method", type=str, default="z_score")
parser.add_argument("--optimizer", type=str, default="RMSProp")
parser.add_argument("--early_stop", type=bool, default=False)
parser.add_argument("--exponential_decay_step", type=int, default=5)
parser.add_argument("--decay_rate", type=float, default=0.5)
parser.add_argument("--dropout_rate", type=float, default=0.5)
parser.add_argument("--leakyrelu_rate", type=int, default=0.2)


args = parser.parse_args()
print(f"Training configs: {args}")
data_file = os.path.join("dataset", args.dataset + ".csv")
data = pd.read_csv(data_file).values

# split data
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[: int(train_ratio * len(data))]
valid_data = data[
    int(train_ratio * len(data)) : int((train_ratio + valid_ratio) * len(data))
]
test_data = data[int((train_ratio + valid_ratio) * len(data)) :]

torch.manual_seed(0)

params = OrderedDict(
    lr=[0.001, 0.0001],
    decay_rate=[0.3, 0.5, 0.7],
    leakyrelu_rate=[0, 0.2],
    horizon=[3, 6, 12],
    window_size=[12, 18, 24, 36],
    device=["cuda"],
)

if __name__ == "__main__":
    try:
        before_tuning = datetime.now().timestamp()
        test_performance = tuning(train_data, valid_data, test_data, args, params)
        after_tuning = datetime.now().timestamp()
        print(f"tuning took {(after_tuning - before_tuning) / 60} minutes")
        pd.DataFrame.from_dict(test_performance).to_csv("performance.csv", index=False)
    except KeyboardInterrupt:
        print("-" * 99)
        print("Exiting from training early")
print("done")
