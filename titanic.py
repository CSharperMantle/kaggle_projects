# Competition: https://www.kaggle.com/competitions/titanic

import random

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn

RAND_SEED = 1234
TRAIN_EPOCHS = 20
TRAIN_ITERS_IN_EPOCH = 2500


class Model(nn.Module):
    def __init__(self, input_dim: int):
        super(Model, self).__init__()
        self.lin_1 = nn.Linear(input_dim, 48)
        self.activation_1 = nn.ReLU()
        self.lin_2 = nn.Linear(48, 48)
        self.activation_2 = nn.ReLU()
        self.lin_3 = nn.Linear(48, 2)
        self.activation_3 = nn.Softmax(dim=1)

    def forward(self, x: t.Tensor):
        x = self.activation_1(self.lin_1(x))
        x = self.activation_2(self.lin_2(x))
        x = self.activation_3(self.lin_3(x))
        return x


def split_train_set(train_set: pd.DataFrame, frac: float) -> (pd.DataFrame, pd.DataFrame):
    another = train_set.sample(frac=frac)
    rest = train_set.drop(another.index)
    return rest, another


def build_logits_ndarray(data: pd.Series) -> np.ndarray:
    data = data.to_numpy()
    mapper = np.vectorize(lambda x: (0, 1) if x else (1, 0))
    data = np.asarray(mapper(data))
    data = np.swapaxes(data, 0, 1)
    return data


def main() -> None:
    t.manual_seed(RAND_SEED)
    random.seed(RAND_SEED)

    train_csv_data = pd.read_csv("dataset/titanic/train_1.csv",
                                 usecols=["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"],
                                 index_col="PassengerId").fillna(-1)
    test_csv_data = pd.read_csv("dataset/titanic/test_1.csv",
                                usecols=["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
                                index_col="PassengerId").fillna(-1)

    train_data, eval_data = split_train_set(train_csv_data, 0.2)

    x_train = train_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y_train = train_data["Survived"]
    x_eval = eval_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y_eval = eval_data["Survived"]

    x_test = test_csv_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

    model = Model(x_train.shape[1]).to(t.device("cuda"), dtype=t.float32)
    optim = t.optim.Adam(model.parameters(), amsgrad=True, lr=0.0001)
    sched = t.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=50, cooldown=10)
    loss_fn = nn.CrossEntropyLoss()
    print(model)

    x_train = t.from_numpy(x_train.to_numpy()).to(t.device("cuda"), dtype=t.float32)
    y_train = t.from_numpy(build_logits_ndarray(y_train)).to(t.device("cuda"), dtype=t.float32)
    x_eval = t.from_numpy(x_eval.to_numpy()).to(t.device("cuda"), dtype=t.float32)
    x_test = t.from_numpy(x_test.to_numpy()).to(t.device("cuda"), dtype=t.float32)
    for i in range(TRAIN_EPOCHS):
        model.train()
        for j in range(TRAIN_ITERS_IN_EPOCH):
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            sched.step(loss)
            print(f"{i + 1},\t{j + 1},\t{loss.item()}")
            optim.zero_grad()
            loss.backward()
            optim.step()

        with t.inference_mode():
            model.eval()
            y_eval_pred = model(x_eval)
            y_eval_pred = [(0 if pair[0] > pair[1] else 1) for pair in y_eval_pred.cpu().numpy()]
            correct_percentage = sum(
                ((1 if y_eval_pred[i] == y_eval.values[i] else 0) for i in range(len(y_eval_pred)))
            ) / len(y_eval_pred)
            print(f"\tEC: {correct_percentage * 100.}%")
            ckpt_path = f"checkpoint/titanic/model-{i}.pth"
            t.save(model, ckpt_path)
            print(f"\tPATH: {ckpt_path}")
            y_test_pred = model(x_test)
            y_test_pred = [(0 if pair[0] > pair[1] else 1) for pair in y_test_pred.cpu().numpy()]
            y_test_pred = pd.DataFrame({"Survived": y_test_pred}, index=test_csv_data.index)
            result_path = f"output/titanic/model-{i}-{int(correct_percentage * 100000)}.csv"
            y_test_pred.to_csv(result_path)


if __name__ == "__main__":
    main()
