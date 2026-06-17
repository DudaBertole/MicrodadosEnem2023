"""
Clustering iterativo do questionário socioeconômico do ENEM
usando One-Hot Encoding + MiniBatchKMeans.

Dependências:
    pip install scikit-learn pandas numpy

Uso:
    python3 enem_kmeans_clustering.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------

COLUNAS_CAT_QST = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025"
]

# Limiar mínimo de melhora do silhouette para considerar como melhora real
SILHOUETTE_THRESHOLD = 0.01

# Número de iterações consecutivas sem melhora antes de parar
PATIENCE = 3

# Número máximo de clusters a tentar (teto de segurança)
K_MAX = 15

# Semente para reprodutibilidade
RANDOM_STATE = 42

# Tamanho da amostra para calcular o silhouette (O(n²), caro no dataset inteiro)
SILHOUETTE_SAMPLE = 10_000


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def encode_features(subset: pd.DataFrame) -> tuple:
    """
    Aplica One-Hot Encoding nas colunas categóricas.
    Retorna a matriz esparsa X e o encoder fitado.
    """
    print("  Aplicando One-Hot Encoding...")
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore", dtype=np.float32)
    X = encoder.fit_transform(subset)
    print(f"  Shape codificada: {X.shape}  (sparsa, {X.nnz} valores não-zero)")
    return X, encoder


def run_kmeans(X, k: int, random_state: int) -> tuple[np.ndarray, float]:
    """
    Executa MiniBatchKMeans para um dado k.

    MiniBatchKMeans usa mini-lotes aleatórios a cada iteração —
    muito mais rápido em datasets grandes com pequena perda de qualidade.

    Retorna:
        labels  : array de rótulos de cluster por amostra
        inertia : soma das distâncias quadráticas ao centróide (↓ melhor)
    """
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
        batch_size=4096,
        max_iter=300,
    )
    labels = model.fit_predict(X)
    return labels, model.inertia_


def compute_metrics(X, labels: np.ndarray, k: int, silhouette_sample: int) -> dict:
    """
    Calcula métricas sobre uma amostra aleatória do dataset.

    Silhouette calculado numa amostra pois é O(n²).
    DB e CH também calculados na amostra para consistência.
    """
    n = X.shape[0]
    sample_size = min(silhouette_sample, n)

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    X_sample = X[sample_idx].toarray()
    labels_sample = labels[sample_idx]

    silhouette = silhouette_score(X_sample, labels_sample)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        davies_bouldin = davies_bouldin_score(X_sample, labels_sample)
        calinski_harabasz = calinski_harabasz_score(X_sample, labels_sample)

    return {
        "k": k,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }


def print_metrics(metrics: dict, inertia: float, best_sil: float | None) -> None:
    """Formata e imprime as métricas de uma iteração."""
    sil = metrics["silhouette"]
    delta = ""
    if best_sil is not None:
        delta = f"  (Δbest = {sil - best_sil:+.4f})"

    print(f"\n  k={metrics['k']}")
    print(f"    Silhouette Score    : {sil:.4f}{delta}")
    print(f"    Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}  (↓ melhor)")
    print(f"    Calinski-Harabasz  : {metrics['calinski_harabasz']:.4f}  (↑ melhor)")
    print(f"    Inércia (KMeans)   : {inertia:.2f}  (↓ melhor)")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    df: pd.DataFrame,
    columns: list[str] = COLUNAS_CAT_QST,
    k_start: int = 2,
    k_max: int = K_MAX,
    threshold: float = SILHOUETTE_THRESHOLD,
    patience: int = PATIENCE,
    random_state: int = RANDOM_STATE,
    silhouette_sample: int = SILHOUETTE_SAMPLE,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Pipeline iterativo de clustering com One-Hot + MiniBatchKMeans.

    Critério de parada por patience: só para após `patience` iterações
    consecutivas sem melhora significativa no silhouette.

    Roda no dataset inteiro — sem necessidade de amostragem para o treino,
    pois MiniBatchKMeans escala para milhões de linhas.
    """
    print("=" * 60)
    print("PIPELINE: Clustering ENEM (One-Hot + MiniBatchKMeans)")
    print("=" * 60)
    print(f"    Patience: {patience} iterações sem melhora")

    # --- Etapa 1: preparar e codificar os dados ---
    print(f"\n[1] Preparando dados...")
    subset = df[columns].astype(str).copy()
    print(f"    Shape original: {subset.shape}")
    X, encoder = encode_features(subset)

    # --- Etapa 2: loop iterativo com patience ---
    print("\n[2] Iniciando clustering iterativo...")
    print(f"    k inicial: {k_start} | patience: {patience} | threshold: {threshold}")
    print(f"    Silhouette calculado em amostra de {silhouette_sample:,} pontos")

    history = []
    best_labels = None
    best_k = k_start
    best_silhouette = -np.inf
    no_improve_count = 0

    for k in range(k_start, k_max + 1):
        print(f"\n{'─' * 40}")
        print(f"  Tentando k = {k}...")

        labels, inertia = run_kmeans(X, k, random_state)
        metrics = compute_metrics(X, labels, k, silhouette_sample)
        metrics["inertia"] = inertia
        metrics["labels"] = labels

        print_metrics(metrics, inertia, best_silhouette if best_labels is not None else None)
        history.append(metrics)

        current_silhouette = metrics["silhouette"]

        if current_silhouette > best_silhouette + threshold:
            best_silhouette = current_silhouette
            best_labels = labels.copy()
            best_k = k
            no_improve_count = 0
            print(f"  ✓  Novo melhor: k={k}, silhouette={current_silhouette:.4f}")
        else:
            no_improve_count += 1
            print(f"  →  Sem melhora significativa ({no_improve_count}/{patience})")

        if no_improve_count >= patience:
            print(f"\n  ⚠  Parada: {patience} iterações sem melhora.")
            print(f"     Melhor k encontrado: k={best_k}.")
            break

    else:
        print(f"\n  ⚠  k_max ({k_max}) atingido. Usando k={best_k}.")

    # --- Etapa 3: consolidar resultados ---
    best_row = next(r for r in history if r["k"] == best_k)
    final_labels = pd.Series(best_row["labels"], index=df.index, name="cluster")

    print("\n" + "=" * 60)
    print(f"RESULTADO FINAL: k = {best_k}")
    print(f"  Silhouette Score    : {best_row['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {best_row['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz  : {best_row['calinski_harabasz']:.4f}")
    print(f"  Inércia             : {best_row['inertia']:.2f}")
    print(f"\nDistribuição dos clusters (dataset completo):")
    print(final_labels.value_counts().sort_index().to_string())
    print("=" * 60)

    history_df = pd.DataFrame([
        {k_: v for k_, v in row.items() if k_ != "labels"}
        for row in history
    ])
    return final_labels, history_df


# ---------------------------------------------------------------------------
# Bloco de execução
# ---------------------------------------------------------------------------

df = pd.read_csv("../data/df_limpo.csv", low_memory=False)

if __name__ == "__main__":
    cluster_labels, metrics_history = run_clustering_pipeline(df)
    df["cluster"] = cluster_labels
    print("\n[INFO] Coluna 'cluster' adicionada ao DataFrame `df`.")
    print("[INFO] Histórico de métricas disponível em `metrics_history`.")