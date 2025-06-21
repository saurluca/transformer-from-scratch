import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
from modules import Sequential, LinearLayer, SGD, MSE, Sigmoid


def train(
    train_data: List[Tuple[np.ndarray, np.ndarray]],
    model: Sequential,
    criterion: MSE,
    optimiser: SGD,
    n_epochs: int = 10,
) -> Tuple[List[float], List[List[np.ndarray]]]:
    train_losses = []
    outputs = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        outputs_epoch = []
        for X, target in train_data:
            # forward pass
            pred = model(X)
            loss = criterion(pred, target)
            train_loss += loss
            outputs_epoch.append(pred)

            # backward pass
            optimiser.zero_grad()
            grad = criterion.backward()
            model.backward(grad)
            optimiser.step()

            # print(f"y {target}, pred {pred}, loss {loss}")
        train_losses.append(train_loss)
        outputs.append(outputs_epoch)
    return train_losses, outputs


def plot_loss(losses):
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_predictions(outputs, targets):
    plt.scatter(targets, outputs)
    plt.title("Predictions vs Targets")
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.show()


def main() -> None:
    np.random.seed(42)

    # config
    n_epochs = 10
    lr = 0.1

    # setup dummy data
    n_samples = 200
    inputs = np.random.uniform(-1, 1, size=(n_samples, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = -0.1
    targets = Sigmoid()._sigmoid(inputs @ true_w + true_b)
    train_data = list(zip(inputs, targets))

    model = Sequential(
        [
            LinearLayer(in_dim=3, out_dim=1),
            Sigmoid(),
            # LinearLayer(in_dim=2, out_dim=1),
            # Sigmoid()
        ]
    )
    criterion = MSE()
    optimiser = SGD(model.params(), criterion, lr=lr, momentum=0.9)

    train_losses, outputs = train(train_data, model, criterion, optimiser, n_epochs)
    plot_loss(train_losses)
    plot_predictions(outputs[-1], targets)

    print(f"final loss {train_losses[-1]}")

    # print out final model params
    final_params = model.params()[0]
    print(
        f"true W {true_w} model w {final_params.W} \n true b {true_b}, model b {final_params.b}"
    )


if __name__ == "main":
    main()


main()
