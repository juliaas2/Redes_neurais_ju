"""Exercicios 2 e 3 de Data.

Uso:
    python docs/exercises/data/code/exercise2_3_analysis.py --csv train.csv

O argumento --csv e obrigatorio para gerar os resultados do Spaceship Titanic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FIGURES = Path(__file__).resolve().parents[1] / "figures"
SEED = 42
N_SAMPLES = 500


def generate_shifted_gaussians(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mean_a = np.zeros(5)
    mean_b = np.full(5, 1.5)
    covariance_a = np.array(
        [[1.0, 0.8, 0.1, 0.0, 0.0], [0.8, 1.0, 0.3, 0.0, 0.0],
         [0.1, 0.3, 1.0, 0.5, 0.0], [0.0, 0.0, 0.5, 1.0, 0.2],
         [0.0, 0.0, 0.0, 0.2, 1.0]]
    )
    covariance_b = np.array(
        [[1.5, -0.7, 0.2, 0.0, 0.0], [-0.7, 1.5, 0.4, 0.0, 0.0],
         [0.2, 0.4, 1.5, 0.6, 0.0], [0.0, 0.0, 0.6, 1.5, 0.3],
         [0.0, 0.0, 0.0, 0.3, 1.5]]
    )
    points_a = rng.multivariate_normal(mean_a, covariance_a, N_SAMPLES)
    points_b = rng.multivariate_normal(mean_b, covariance_b, N_SAMPLES)
    return np.vstack([points_a, points_b]), np.repeat([0, 1], N_SAMPLES)


def generate_shells(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    directions = rng.normal(size=(2 * N_SAMPLES, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.concatenate([
        rng.normal(2.0, 0.4, N_SAMPLES),
        rng.normal(5.0, 0.4, N_SAMPLES),
    ])
    return directions * radii[:, None], np.repeat([0, 1], N_SAMPLES)


def plot_pca(datasets: list[tuple[np.ndarray, np.ndarray]], path: Path) -> list[float]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    explained = []
    for ax, (X, y), title in zip(axes, datasets, ["Dataset I: gaussianas", "Dataset II: cascas"]):
        pca = PCA(n_components=2)
        projection = pca.fit_transform(X)
        explained.append(float(pca.explained_variance_ratio_.sum()))
        for label, color, name in [(0, "#2878b5", "Classe A / C"), (1, "#d1495b", "Classe B / D")]:
            ax.scatter(projection[y == label, 0], projection[y == label, 1], s=12, alpha=0.55, color=color, label=name)
        ax.set_title(title)
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return explained


def plot_radius_histograms(datasets: list[tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (X, y), title in zip(axes, datasets, ["Dataset I", "Dataset II"]):
        radius = np.linalg.norm(X, axis=1)
        ax.hist(radius[y == 0], bins=28, alpha=0.65, label="Classe 0 / C", color="#2878b5")
        ax.hist(radius[y == 1], bins=28, alpha=0.65, label="Classe 1 / D", color="#d1495b")
        ax.set_title(f"Figura 5 — raios: {title}")
        ax.set_xlabel("Raio ||x||")
        ax.set_ylabel("Frequência")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def exercise_two() -> None:
    rng = np.random.default_rng(SEED)
    shifted = generate_shifted_gaussians(rng)
    shells = generate_shells(rng)
    FIGURES.mkdir(parents=True, exist_ok=True)
    explained = plot_pca([shifted, shells], FIGURES / "fig04-pca.png")
    plot_radius_histograms([shifted, shells], FIGURES / "fig05-radius-histograms.png")
    for name, (X, y) in [("Dataset I", shifted), ("Dataset II", shells)]:
        centers = [X[y == label].mean(axis=0) for label in [0, 1]]
        print(f"{name}: distancia entre centros = {np.linalg.norm(centers[0] - centers[1]):.6f}")
    print(f"Dataset I: variancia explicada PC1+PC2 = {explained[0]:.6%}")
    print(f"Dataset II: variancia explicada PC1+PC2 = {explained[1]:.6%}")
    print(f"Dataset II: raios medios = {np.linalg.norm(shells[0][shells[1] == 0], axis=1).mean():.6f}, {np.linalg.norm(shells[0][shells[1] == 1], axis=1).mean():.6f}")


def exercise_three(csv_path: Path) -> None:
    data = pd.read_csv(csv_path)
    target = data["Transported"].astype(bool)
    print(f"Balance positivo (Transported=True): {target.mean():.6%}")
    print("Valores ausentes:")
    missing = pd.DataFrame({"count": data.isna().sum(), "percent": data.isna().mean() * 100})
    print(missing.to_string())
    spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    print("Estatisticas de gastos:")
    print(data[spend].agg(["mean", "median", "max"]).T.to_string())

    features = data.drop(columns="Transported").copy()
    features["TotalSpend"] = features[spend].sum(axis=1, min_count=1)
    features = features.drop(columns=["Cabin", "Name", "PassengerId"])
    features[spend + ["TotalSpend"]] = np.log1p(features[spend + ["TotalSpend"]])
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED
    )
    categorical = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    numerical = [column for column in features.columns if column not in categorical]
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numerical", numeric_steps, numerical),
        ("categorical", categorical_steps, categorical),
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    before = X_train["FoodCourt"].dropna()
    after = np.log1p(X_train["FoodCourt"].fillna(X_train["FoodCourt"].median()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(before, bins=30, color="#2878b5")
    axes[0].set_title("Figura 6 — FoodCourt antes")
    axes[1].hist(after, bins=30, color="#d1495b")
    axes[1].set_title("Figura 6 — FoodCourt depois de log(1+x)")
    for ax in axes:
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frequência")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-foodcourt-before-after.png", dpi=150)
    plt.close(fig)
    print(f"Treino: {X_train_processed.shape}; teste: {X_test_processed.shape}")
    print(f"Faixa treino: [{X_train_processed.min():.6f}, {X_train_processed.max():.6f}]")
    print(f"Faixa teste: [{X_test_processed.min():.6f}, {X_test_processed.max():.6f}]")
    print(f"NaN no treino/teste: {np.isnan(X_train_processed).sum()}/{np.isnan(X_test_processed).sum()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="caminho para o train.csv do Spaceship Titanic")
    args = parser.parse_args()
    exercise_two()
    if args.csv is None:
        print("Exercicio 3: informe --csv train.csv para executar o preprocessamento.")
    else:
        exercise_three(args.csv)


if __name__ == "__main__":
    main()
