"""
Clustering iterativo do questionário socioeconômico do ENEM
usando K-Modes (distância de Hamming, centros por moda).

Dependências:
    pip install kmodes scikit-learn pandas numpy

Uso:
    python3 enem_kmodes_clustering.py
"""

import warnings
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------

COLUNAS_CAT_QST = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025",
    "SG_UF_PROVA",
    "TP_ESCOLA",
    "TP_ST_CONCLUSAO",
    "TP_FAIXA_ETARIA",
    "TP_COR_RACA",
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

def prepare_data(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepara os dados em dois formatos:

    - X_str : array de strings para o KModes (trabalha com categorias originais)
    - X_num : array numérico com LabelEncoding para as métricas de avaliação
              (silhouette, DB, CH exigem representação numérica)
    """
    print(f"  Selecionando {len(columns)} colunas...")
    subset = df[columns].astype(str).copy()
    print(f"  Shape: {subset.shape}")

    X_str = subset.values

    # LabelEncoder por coluna: cada categoria vira um inteiro
    X_num = np.zeros_like(X_str, dtype=np.int32)
    for i, col in enumerate(subset.columns):
        le = LabelEncoder()
        X_num[:, i] = le.fit_transform(X_str[:, i])

    return X_str, X_num


def run_kmodes(X_str: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, float]:
    """
    Executa K-Modes para um dado k.

    Inicialização 'Huang' é mais eficiente que aleatória para dados categóricos.
    n_init=5 roda 5 vezes e retém o menor custo, reduzindo sensibilidade
    à inicialização.

    Retorna:
        labels : array de rótulos de cluster por amostra
        cost   : soma das distâncias de Hamming ao modo (↓ melhor)
    """
    model = KModes(
        n_clusters=k,
        init="Huang",
        n_init=5,
        random_state=random_state,
        verbose=0,
    )
    labels = model.fit_predict(X_str)
    return labels, model.cost_


def compute_metrics(X_num: np.ndarray, labels: np.ndarray, k: int, silhouette_sample: int) -> dict:
    """
    Calcula métricas sobre uma amostra aleatória do dataset.

    Silhouette usa distância de Hamming normalizada — consistente com
    a distância usada internamente pelo K-Modes.
    DB e CH usam representação numérica (LabelEncoded) como aproximação.
    """
    n = X_num.shape[0]
    sample_size = min(silhouette_sample, n)

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    X_sample = X_num[sample_idx]
    labels_sample = labels[sample_idx]

    silhouette = silhouette_score(X_sample, labels_sample, metric="hamming")

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


def print_metrics(metrics: dict, cost: float, best_sil: float | None) -> None:
    """Formata e imprime as métricas de uma iteração."""
    sil = metrics["silhouette"]
    delta = ""
    if best_sil is not None:
        delta = f"  (Δbest = {sil - best_sil:+.4f})"

    print(f"\n  k={metrics['k']}")
    print(f"    Silhouette Score    : {sil:.4f}{delta}")
    print(f"    Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}  (↓ melhor)")
    print(f"    Calinski-Harabasz  : {metrics['calinski_harabasz']:.4f}  (↑ melhor)")
    print(f"    Custo (Hamming)    : {cost:.2f}  (↓ melhor)")


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
    Pipeline iterativo de clustering com K-Modes.

    Critério de parada por patience: só para após `patience` iterações
    consecutivas sem melhora significativa no silhouette.

    Roda no dataset inteiro — sem necessidade de amostragem para o treino,
    pois K-Modes escala bem para milhões de linhas.
    """
    print("=" * 60)
    print("PIPELINE: Clustering ENEM (K-Modes)")
    print("=" * 60)
    print(f"    Patience: {patience} iterações sem melhora")

    # --- Etapa 1: preparar os dados ---
    print(f"\n[1] Preparando dados...")
    X_str, X_num = prepare_data(df, columns)

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

        labels, cost = run_kmodes(X_str, k, random_state)
        metrics = compute_metrics(X_num, labels, k, silhouette_sample)
        metrics["cost"] = cost
        metrics["labels"] = labels

        print_metrics(metrics, cost, best_silhouette if best_labels is not None else None)
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
    print(f"  Custo (Hamming)    : {best_row['cost']:.2f}")
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