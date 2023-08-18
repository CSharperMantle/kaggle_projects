# Competition: https://www.kaggle.com/competitions/digit-recognizer

import random
import time

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

RAND_SEED = 1234
BATCH_SIZE = 64
TRAIN_EPOCHS = 20
TRAIN_ROUNDS_IN_EPOCH = 1000


class Model(nn.Module):
    def __init__(self, img_size: int):
        super(Model, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)

        self.fc_1 = nn.Linear(img_size ** 2 * 64, 128)
        self.bn_3 = nn.BatchNorm1d(128)

        self.fc_2 = nn.Linear(128, 10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv_1(x)
        x = F.leaky_relu(self.bn_1(x), 0.1)

        x = self.conv_2(x)
        x = F.leaky_relu(self.bn_2(x), 0.1)

        x = t.flatten(x, 1)
        x = self.fc_1(x)
        x = F.leaky_relu(self.bn_3(x), 0.1)

        x = F.softmax(self.fc_2(x), dim=1)
        return x


def chunks(l: list, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def build_logits(data: pd.Series) -> np.ndarray:
    logits = np.zeros((len(data), 10))
    for i, tag in enumerate(data):
        logits[i][tag] = 1
    return logits


def split_train_set(train_set: pd.DataFrame, frac: float) -> (pd.DataFrame, pd.DataFrame):
    another = train_set.sample(frac=frac)
    rest = train_set.drop(another.index)
    return rest, another


def main() -> None:
    t.manual_seed(RAND_SEED)
    random.seed(RAND_SEED)

    train_csv_data = pd.read_csv("dataset/digit_recognizer/train.csv")
    train_data, eval_data = split_train_set(train_csv_data, 0.1)
    test_csv_data = pd.read_csv("dataset/digit_recognizer/test.csv")

    test_batch_mats = np.reshape(test_csv_data.iloc[0:64, :].to_numpy(), (-1, 28, 28))

    model = Model(28)
    summary(model, input_size=(BATCH_SIZE, 1, 28, 28))

    optimizer = t.optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, cooldown=3)
    for i in range(TRAIN_EPOCHS):
        model.train()
        for j in range(TRAIN_ROUNDS_IN_EPOCH):
            eval_batch = train_data.sample(BATCH_SIZE)
            batch_mats = np.reshape(eval_batch.iloc[:, 1:].to_numpy(), (-1, 28, 28))
            batch_tags = build_logits(eval_batch.iloc[:, 0])

            batch_mats = t.from_numpy(batch_mats).unsqueeze(1).float().to("cuda", dtype=t.float32)
            batch_tags = t.from_numpy(batch_tags).float().to("cuda", dtype=t.float32)

            pred = model(batch_mats)
            loss = loss_fn(pred, batch_tags)
            scheduler.step(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {i + 1} | Round {j + 1} | Loss: {loss.item():.4f}")

        with t.inference_mode():
            model.eval()
            eval_batch = eval_data.sample(BATCH_SIZE)
            eval_batch_mats = np.reshape(eval_batch.iloc[:, 1:].to_numpy(), (-1, 28, 28))
            eval_batch_tags = eval_batch.iloc[:, 0].to_numpy()

            eval_batch_mats = t.from_numpy(eval_batch_mats).unsqueeze(1).float().to("cuda", dtype=t.float32)

            eval_pred = model(eval_batch_mats)

            eval_pred = eval_pred.cpu().detach().numpy()
            accuracy = sum(
                (1 if np.argmax(eval_pred[i]) == eval_batch_tags[i] else 0) for i in range(BATCH_SIZE)
            ) / BATCH_SIZE
            print(f"Epoch {i + 1} | Accuracy: {accuracy * 100:.2f}%")

            ckpt_path = f"checkpoint/digit_recognizer/model-{i}.pth"
            t.save(model, ckpt_path)
            print(f"\tPath: {ckpt_path}")

            test_pred = pd.DataFrame({"ImageId": [], "Label": []})
            for chunk in chunks([i for i in range(0, len(test_csv_data))], BATCH_SIZE):
                batch_mats = np.reshape(test_csv_data.iloc[chunk, :].to_numpy(), (-1, 28, 28))
                batch_mats = t.from_numpy(batch_mats).unsqueeze(1).float().to("cuda", dtype=t.float32)
                batch_pred = model(batch_mats)
                batch_pred = [np.argmax(vec) for vec in batch_pred.cpu().detach().numpy()]
                batch_pred = pd.DataFrame(
                    {"ImageId": map(lambda x: x + 1, chunk), "Label": batch_pred}
                )
                test_pred = pd.concat([test_pred.astype(int), batch_pred.astype(int)])

            result_path = f"output/digit_recognizer/model-{i + 1}-{int(accuracy * 10000)}.csv"
            test_pred.to_csv(result_path, index=False)

        time.sleep(0.5)


if __name__ == "__main__":
    main()
