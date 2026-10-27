"""Exercises 2–4 — MLP from scratch (ReLU hidden, SGD).

sklearn entra só para gerar/splitar os dados, a regressão logística
de baseline e a matriz de confusão. Forward, loss, gradiente e update
são os mesmos métodos em todos os exercícios.

Uso (na raiz do repositório):

    python docs/exercises/mlp/code/mlp.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)


class MLP:
    """MLP de profundidade arbitrária. ``sizes`` é a lista [d_in, h1, ..., d_out].

    Camadas ocultas: ReLU. Saída: sigmoide (binário) ou softmax (multiclasse).
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator, output: str):
        self.sizes = list(sizes)
        self.output = output
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0 / n_in)
            self.W.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache = {"a0": X}
        a = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            self.cache[f"z{i+1}"] = z
            if i == last:
                a = _sigmoid(z) if self.output == "binary" else _softmax(z)
            else:
                a = np.maximum(0.0, z)
            self.cache[f"a{i+1}"] = a
        return a

    def loss(self, yhat: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        if self.output == "binary":
            y = y.reshape(-1, 1)
            return float(-np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps)))
        return float(-np.mean(np.log(yhat[np.arange(len(y)), y] + eps)))

    def backward(self, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # BCE+sigmoide e CCE+softmax compartilham dL/du = (ŷ − y) / N.
        n = self.cache["a0"].shape[0]
        L = len(self.W)
        yhat = self.cache[f"a{L}"]
        if self.output == "binary":
            dz = (yhat - y.reshape(-1, 1)) / n
        else:
            y_oh = np.zeros_like(yhat)
            y_oh[np.arange(n), y] = 1.0
            dz = (yhat - y_oh) / n

        dW, db = [None] * L, [None] * L
        for i in reversed(range(L)):
            a_prev = self.cache[f"a{i}"]
            dW[i] = a_prev.T @ dz
            db[i] = dz.sum(axis=0)
            if i > 0:
                da = dz @ self.W[i].T
                dz = da * (self.cache[f"z{i}"] > 0)

        return dW, db

    def update(self, dW: list[np.ndarray], db: list[np.ndarray], lr: float) -> None:
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        yhat = self.forward(X)
        if self.output == "binary":
            return (yhat.ravel() >= 0.5).astype(int)
        return yhat.argmax(axis=1)

    def n_params(self) -> int:
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[float]:
    losses: list[float] = []
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        Xs, ys = X[order], y[order]
        for start in range(0, n, batch_size):
            xb = Xs[start : start + batch_size]
            yb = ys[start : start + batch_size]
            model.forward(xb)
            dW, db = model.backward(yb)
            model.update(dW, db, lr)
        yhat = model.forward(X)
        losses.append(model.loss(yhat, y))
    return losses


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(X) == y))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def plot_loss(ax, losses, title: str, color="#4C78A8") -> None:
    ax.plot(range(1, len(losses) + 1), losses, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (treino)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ----- Exercise 2 -----
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X2, c = make_blobs(
        n_samples=[500, 250, 250], centers=centers, cluster_std=1.2, random_state=42
    )
    y2 = (c > 0).astype(int)
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(*X2[y2 == 0].T, s=14, alpha=0.75, color="#4C78A8", label="Classe 0")
    ax.scatter(*X2[y2 == 1].T, s=14, alpha=0.75, color="#F58518", label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 1 — blobs: classe 1 em lados opostos")
    ax.legend()
    save(fig, "fig01-blobs.png")

    lin2 = LogisticRegression(random_state=42, max_iter=1000).fit(X2_tr, y2_tr)
    acc_lin2 = float(lin2.score(X2_te, y2_te))
    print(f"Ex2 linear baseline test acc = {acc_lin2:.6f}")

    sizes2 = [2, 16, 1]
    lr2, epochs2, bs2 = 0.08, 800, 64
    mlp2 = MLP(sizes2, RNG, output="binary")
    loss2 = train(mlp2, X2_tr, y2_tr, lr=lr2, epochs=epochs2, batch_size=bs2, rng=RNG)
    acc2 = accuracy(mlp2, X2_te, y2_te)
    print(f"Ex2 MLP {sizes2}  lr={lr2}  epochs={epochs2}  test acc={acc2:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_loss(ax, loss2, "Figura 2 — loss de treino (Exercise 2)")
    save(fig, "fig02-loss-ex2.png")

    pad = 1.0
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = mlp2.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#4C78A8", "#F58518"], alpha=0.25)
    ax.scatter(*X2_te[y2_te == 0].T, s=18, color="#4C78A8", label="Classe 0 (teste)", edgecolors="k", linewidths=0.3)
    ax.scatter(*X2_te[y2_te == 1].T, s=18, color="#F58518", label="Classe 1 (teste)", edgecolors="k", linewidths=0.3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figura 3 — fronteira do MLP no conjunto de teste")
    ax.legend()
    save(fig, "fig03-boundary-ex2.png")

    # ----- Exercise 3 -----
    X3, y3 = make_classification(
        n_samples=1500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=42,
    )
    X3_tr, X3_te, y3_tr, y3_te = train_test_split(
        X3, y3, test_size=0.2, random_state=42, stratify=y3
    )

    lin3 = LogisticRegression(random_state=42, max_iter=2000).fit(X3_tr, y3_tr)
    acc_lin3 = float(lin3.score(X3_te, y3_te))
    print(f"Ex3 linear baseline test acc = {acc_lin3:.6f}")

    sizes3 = [4, 32, 3]
    lr3, epochs3, bs3 = 0.08, 400, 64
    mlp3 = MLP(sizes3, RNG, output="multiclass")
    loss3 = train(mlp3, X3_tr, y3_tr, lr=lr3, epochs=epochs3, batch_size=bs3, rng=RNG)
    acc3 = accuracy(mlp3, X3_te, y3_te)
    print(f"Ex3 MLP {sizes3}  params={mlp3.n_params()}  test acc={acc3:.6f}  final train loss={loss3[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_loss(ax, loss3, "Figura 4 — loss de treino (Exercise 3)")
    save(fig, "fig04-loss-ex3.png")

    cm = confusion_matrix(y3_te, mlp3.predict(X3_te))
    print("confusion matrix\n", cm)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    print(f"most confused pair: true {i} predicted {j}  count={off[i, j]}")

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xlabel("predito")
    ax.set_ylabel("verdadeiro")
    ax.set_title("Figura 5 — matriz de confusão (teste)")
    for r in range(3):
        for col in range(3):
            ax.text(col, r, str(cm[r, col]), ha="center", va="center", color="black")
    save(fig, "fig05-confusion.png")

    # ----- Exercise 4 -----
    sizes4 = [4, 32, 32, 3]
    mlp4 = MLP(sizes4, RNG, output="multiclass")
    loss4 = train(mlp4, X3_tr, y3_tr, lr=lr3, epochs=epochs3, batch_size=bs3, rng=RNG)
    acc4 = accuracy(mlp4, X3_te, y3_te)
    print(f"Ex4 MLP {sizes4}  params={mlp4.n_params()}  test acc={acc4:.6f}  final train loss={loss4[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, len(loss3) + 1), loss3, color="#4C78A8", label="Exercise 3 (1 hidden)")
    ax.plot(range(1, len(loss4) + 1), loss4, color="#E45756", label="Exercise 4 (2 hidden)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (treino)")
    ax.set_title("Figura 6 — loss de treino: 1 vs. 2 camadas ocultas")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig06-depth-loss.png")

    accs3, accs4 = [], []
    for _ in range(3):
        m3 = MLP(sizes3, RNG, output="multiclass")
        train(m3, X3_tr, y3_tr, lr=lr3, epochs=epochs3, batch_size=bs3, rng=RNG)
        accs3.append(accuracy(m3, X3_te, y3_te))
        m4 = MLP(sizes4, RNG, output="multiclass")
        train(m4, X3_tr, y3_tr, lr=lr3, epochs=epochs3, batch_size=bs3, rng=RNG)
        accs4.append(accuracy(m4, X3_te, y3_te))
    print(f"Ex3 extra seeds acc={accs3}  mean={np.mean(accs3):.6f}")
    print(f"Ex4 extra seeds acc={accs4}  mean={np.mean(accs4):.6f}")
    print(f"mean(Ex4) - mean(Ex3) = {np.mean(accs4) - np.mean(accs3):.6f}")


if __name__ == "__main__":
    main()
