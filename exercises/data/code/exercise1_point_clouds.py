"""Exercise 1 — Geometry and Spread in 2D.

Gera as 4 classes gaussianas do enunciado, mede separation ratio e mixing rate
para vários fatores de escala s, e salva as figuras em ``figures/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

FIGURES = Path(__file__).resolve().parents[1] / "figures"
SEED = 42

# Médias nunca mudam; só o desvio é multiplicado por s.
CLASSES = {
    0: {"mean": np.array([2.0, 3.0]), "std": np.array([0.8, 2.5])},
    1: {"mean": np.array([5.0, 6.0]), "std": np.array([1.2, 1.9])},
    2: {"mean": np.array([8.0, 1.0]), "std": np.array([0.9, 0.9])},
    3: {"mean": np.array([15.0, 4.0]), "std": np.array([0.5, 2.0])},
}
N_PER_CLASS = 100
SCALES = (0.5, 1.0, 2.0, 4.0)
PAIRS = [(i, j) for i in CLASSES for j in CLASSES if i < j]


def sigma_bar(std: np.ndarray) -> float:
    """Dispersão média da classe: (σ_x + σ_y) / 2."""
    return float(std.mean())


def separation_ratio(i: int, j: int, scale: float = 1.0) -> float:
    """r_ij = ||μ_i − μ_j|| / (σ̄_i + σ̄_j), com desvios já escalados por s."""
    mu_i, mu_j = CLASSES[i]["mean"], CLASSES[j]["mean"]
    si = sigma_bar(CLASSES[i]["std"] * scale)
    sj = sigma_bar(CLASSES[j]["std"] * scale)
    return float(np.linalg.norm(mu_i - mu_j) / (si + sj))


def unit_noise(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Ruído N(0, 1) fixo por classe — o mesmo para todo s, só a escala muda."""
    return {c: rng.standard_normal((N_PER_CLASS, 2)) for c in CLASSES}


def assemble(noise: dict[int, np.ndarray], scale: float) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label, params in CLASSES.items():
        xs.append(params["mean"] + scale * params["std"] * noise[label])
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Fração de pontos cujo centro mais próximo não é o da própria classe."""
    centers = np.stack([CLASSES[c]["mean"] for c in CLASSES])
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    return float((nearest != y).mean())


def plot_clouds(ax, X: np.ndarray, y: np.ndarray, title: str) -> None:
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for c in CLASSES:
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.75, color=colors[c], label=f"Classe {c}")
        mu = CLASSES[c]["mean"]
        ax.scatter(*mu, marker="X", s=90, color=colors[c], edgecolors="black", linewidths=0.6, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def plot_decision(ax, clf, X: np.ndarray, y: np.ndarray, title: str, xlim, ylim) -> None:
    xx, yy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.arange(-0.5, 4.5, 1), alpha=0.25, colors=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plot_clouds(ax, X, y, title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    noise = unit_noise(rng)
    datasets = {s: assemble(noise, s) for s in SCALES}

    # --- Figura 1: s = 1 ---
    X1, y1 = datasets[1.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_clouds(ax, X1, y1, "Figura 1 — nuvens gaussianas ($s = 1$)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
    plt.close(fig)

    # --- Figura 2: quatro escalas, mesmos eixos ---
    all_X = np.vstack([datasets[s][0] for s in SCALES])
    pad = 0.8
    xlim = (all_X[:, 0].min() - pad, all_X[:, 0].max() + pad)
    ylim = (all_X[:, 1].min() - pad, all_X[:, 1].max() + pad)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), SCALES):
        Xs, ys = datasets[s]
        plot_clouds(ax, Xs, ys, f"$s = {s}$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-spread-scales.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Tabela r_ij em s = 1 e predição em s = 2 ---
    print("Separation ratios em s = 1 (parâmetros do enunciado, sem amostragem):")
    ratios = {}
    for i, j in PAIRS:
        r = separation_ratio(i, j, scale=1.0)
        ratios[(i, j)] = r
        print(f"  r_{i}{j} = {r:.4f}")
    smallest_pair = min(ratios, key=ratios.get)
    print(f"Menor: r_{smallest_pair[0]}{smallest_pair[1]} = {ratios[smallest_pair]:.4f}")
    print(f"Em s = 2 o mesmo par vale {ratios[smallest_pair] / 2:.4f} (r_ij escala com 1/s).")

    # --- Mixing rate × s ---
    mix = {s: mixing_rate(*datasets[s]) for s in SCALES}
    print("Mixing rate:")
    for s, m in mix.items():
        print(f"  s={s:>3} | mixing = {100 * m:.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(SCALES), [mix[s] for s in SCALES], "o-", color="#4C78A8")
    ax.set_xlabel("fator de escala $s$")
    ax.set_ylabel("mixing rate")
    ax.set_title("Figura 3 — taxa de mistura $\\times$ $s$")
    ax.set_xticks(SCALES)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-mixing-rate.png", dpi=150)
    plt.close(fig)

    # --- Fronteiras de um MLP (esboço da parte C) ---
    clf = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation="relu",
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(X1, y1)

    X4, y4 = datasets[4.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    plot_decision(axes[0], clf, X1, y1, "MLP treinado em $s = 1$", xlim, ylim)
    pred4 = clf.predict(X4)
    plot_decision(axes[1], clf, X4, y4, f"mesmo MLP em $s = 4$ (erros = {(pred4 != y4).mean():.1%})", xlim, ylim)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-nn-boundaries.png", dpi=150)
    plt.close(fig)

    print(f"Acurácia do MLP em s=1 (treino): {(clf.predict(X1) == y1).mean():.3f}")
    print(f"Acurácia do mesmo MLP em s=4: {(pred4 == y4).mean():.3f}")


if __name__ == "__main__":
    main()
