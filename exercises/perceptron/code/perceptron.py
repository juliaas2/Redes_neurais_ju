"""Exercise 1–2 — Perceptron from scratch (labels in {0, 1}).

A mesma semente alimenta geração de dados e inicialização dos pesos.
Uso (na raiz do repositório):

    python docs/exercises/perceptron/code/perceptron.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)
ETA = 0.01
MAX_EPOCHS = 100
N_PER_CLASS = 1000


def generate(mean0, mean1, cov) -> tuple[np.ndarray, np.ndarray]:
    """1000 pontos por classe, rótulos 0 e 1."""
    cov = np.asarray(cov, dtype=float)
    X0 = RNG.multivariate_normal(mean0, cov, N_PER_CLASS)
    X1 = RNG.multivariate_normal(mean1, cov, N_PER_CLASS)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


def step(z: np.ndarray | float) -> np.ndarray | int:
    """step(z) = 1 se z >= 0, senão 0."""
    return (np.asarray(z) >= 0).astype(int)


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return step(X @ w + b)


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.mean(predict(X, w, b) == y))


def train(
    X: np.ndarray,
    y: np.ndarray,
    eta: float,
    w: np.ndarray,
    b: float,
    pocket: bool = False,
) -> dict:
    """Uma passagem por amostra: ŷ = step(w·x+b); w += η(y-ŷ)x; b += η(y-ŷ).

    Para quando um epoch inteiro não atualiza, ou ao chegar em 100 epochs.
    Com ``pocket=True``, guarda a melhor (w, b) vista após cada atualização.
    """
    w = np.array(w, dtype=float, copy=True)
    b = float(b)
    acc_hist = [accuracy(X, y, w, b)]
    pocket_hist = [acc_hist[0]]
    w_pocket, b_pocket = w.copy(), b
    best_acc = acc_hist[0]
    best_epoch = 0
    n_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        updates = 0
        for x_i, y_i in zip(X, y):
            y_hat = int(step(float(w @ x_i + b)))
            err = y_i - y_hat
            if err == 0:
                continue
            w = w + eta * err * x_i
            b = b + eta * err
            updates += 1
            if pocket:
                acc_now = accuracy(X, y, w, b)
                if acc_now > best_acc:
                    best_acc = acc_now
                    w_pocket, b_pocket = w.copy(), b
                    best_epoch = epoch
        n_epochs = epoch
        acc_hist.append(accuracy(X, y, w, b))
        if pocket:
            pocket_hist.append(best_acc)
        if updates == 0:
            break

    return {
        "w": w,
        "b": b,
        "epochs": n_epochs,
        "acc": acc_hist[-1],
        "acc_hist": acc_hist,
        "w_pocket": w_pocket,
        "b_pocket": b_pocket,
        "acc_pocket": best_acc,
        "pocket_epoch": best_epoch,
        "pocket_hist": pocket_hist,
    }


def scatter_classes(ax, X, y, title: str) -> None:
    ax.scatter(*X[y == 0].T, s=12, alpha=0.7, label="Classe 0", color="#4C78A8")
    ax.scatter(*X[y == 1].T, s=12, alpha=0.7, label="Classe 1", color="#F58518")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()


def draw_boundary(ax, w, b, xlim, ylim, **kwargs) -> None:
    """Reta w·x + b = 0."""
    if abs(w[1]) > 1e-12:
        xs = np.linspace(*xlim, 200)
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, **kwargs)
    else:
        x = -b / w[0]
        ax.axvline(x, **kwargs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def mark_errors(ax, X, y, w, b) -> None:
    wrong = predict(X, w, b) != y
    if wrong.any():
        ax.scatter(
            *X[wrong].T,
            s=36,
            facecolors="none",
            edgecolors="#E45756",
            linewidths=1.1,
            label="erro",
            zorder=5,
        )


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def fmt_w(w, b) -> str:
    return f"w = [{w[0]:.6f}, {w[1]:.6f}], b = {b:.6f}"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cov_sep = [[0.5, 0.0], [0.0, 0.5]]
    cov_ovl = [[1.5, 0.0], [0.0, 1.5]]

    # --- Exercise 1 ---
    X1, y1 = generate([1.5, 1.5], [5.0, 5.0], cov_sep)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter_classes(ax, X1, y1, "Figura 1 — dados linearmente separáveis")
    save(fig, "fig01-data-separable.png")

    w0 = RNG.normal(0, 0.01, size=2)
    b0 = 0.0
    print("Init Exercise 1:", fmt_w(w0, b0))

    r001 = train(X1, y1, eta=ETA, w=w0, b=b0, pocket=False)
    r100 = train(X1, y1, eta=1.0, w=w0, b=b0, pocket=False)

    pad = 0.8
    xlim1 = (X1[:, 0].min() - pad, X1[:, 0].max() + pad)
    ylim1 = (X1[:, 1].min() - pad, X1[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter_classes(ax, X1, y1, "Figura 2 — fronteira ($\\eta=0.01$)")
    draw_boundary(ax, r001["w"], r001["b"], xlim1, ylim1, color="black", lw=2, label="fronteira")
    mark_errors(ax, X1, y1, r001["w"], r001["b"])
    ax.legend()
    save(fig, "fig02-boundary-ex1.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(len(r001["acc_hist"])), r001["acc_hist"], "o-", color="#4C78A8")
    ax.set_xlabel("epoch (0 = inicialização)")
    ax.set_ylabel("acurácia")
    ax.set_title("Figura 3 — acurácia $\\times$ epoch (Exercise 1)")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    save(fig, "fig03-accuracy-ex1.png")

    u001 = r001["w"] / np.linalg.norm(r001["w"])
    u100 = r100["w"] / np.linalg.norm(r100["w"])

    print("--- Exercise 1 ---")
    print("eta=0.01", fmt_w(r001["w"], r001["b"]))
    print(f"epochs={r001['epochs']}  acc={r001['acc']:.6f}")
    print("eta=1.00", fmt_w(r100["w"], r100["b"]))
    print(f"epochs={r100['epochs']}  acc={r100['acc']:.6f}")
    print(f"w/||w|| eta=0.01 = [{u001[0]:.6f}, {u001[1]:.6f}]")
    print(f"w/||w|| eta=1.00 = [{u100[0]:.6f}, {u100[1]:.6f}]")
    print(f"cos entre direções = {float(u001 @ u100):.6f}")

    # --- Exercise 2 ---
    X2, y2 = generate([3.0, 3.0], [4.0, 4.0], cov_ovl)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter_classes(ax, X2, y2, "Figura 4 — dados com sobreposição")
    save(fig, "fig04-data-overlap.png")

    w2 = RNG.normal(0, 0.01, size=2)
    b2 = 0.0
    print("Init Exercise 2:", fmt_w(w2, b2))
    r2 = train(X2, y2, eta=ETA, w=w2, b=b2, pocket=True)

    xlim2 = (X2[:, 0].min() - pad, X2[:, 0].max() + pad)
    ylim2 = (X2[:, 1].min() - pad, X2[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scatter_classes(ax, X2, y2, "Figura 5 — fronteiras final e pocket")
    draw_boundary(ax, r2["w"], r2["b"], xlim2, ylim2, color="#E45756", lw=2, ls="--", label="final")
    draw_boundary(
        ax, r2["w_pocket"], r2["b_pocket"], xlim2, ylim2, color="black", lw=2, label="pocket"
    )
    mark_errors(ax, X2, y2, r2["w_pocket"], r2["b_pocket"])
    ax.legend()
    save(fig, "fig05-boundary-ex2.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    epochs = range(len(r2["acc_hist"]))
    ax.plot(epochs, r2["acc_hist"], "-", color="#E45756", label="pesos atuais")
    ax.plot(epochs, r2["pocket_hist"], "-", color="#4C78A8", label="pocket (melhor até agora)")
    ax.set_xlabel("epoch (0 = inicialização)")
    ax.set_ylabel("acurácia")
    ax.set_title("Figura 6 — acurácia atual vs. pocket (Exercise 2)")
    ax.set_ylim(0.35, 0.85)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig06-accuracy-ex2.png")

    print("--- Exercise 2 ---")
    print("final ", fmt_w(r2["w"], r2["b"]), f"acc={r2['acc']:.6f}")
    print("pocket", fmt_w(r2["w_pocket"], r2["b_pocket"]), f"acc={r2['acc_pocket']:.6f}")
    print(f"pocket_epoch={r2['pocket_epoch']}  ran_epochs={r2['epochs']}")
    print(f"||w_final||={np.linalg.norm(r2['w']):.4f}  b_final={r2['b']:.4f}")
    print(f"mean ||x||={np.linalg.norm(X2, axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
