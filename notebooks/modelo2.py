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

# Colunas do questionário socioeconômico (apenas categóricas)
COLUNAS_CAT_QST = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025"
]

# Limiar mínimo de melhora do coeficiente de silhueta para continuar iterando
SILHOUETTE_THRESHOLD = 0.01

# Número máximo de clusters a tentar (teto de segurança)
K_MAX = 15

# Semente para reprodutibilidade
RANDOM_STATE = 42

# Tamanho da amostra usada para calcular o silhouette (caro no dataset inteiro)
SILHOUETTE_SAMPLE = 10_000


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def encode_features(subset: pd.DataFrame) -> tuple:
    """
    Aplica One-Hot Encoding nas colunas categóricas.

    Retorna:
        X       : matriz esparsa (n x n_categorias_total)
        encoder : objeto OneHotEncoder fitado (usado para transformar novos dados)

    One-Hot representa cada categoria como uma coluna binária (0/1).
    Com 25 questões e ~5-8 categorias cada, X terá ~150-200 colunas.
    A matriz é mantida esparsa para economizar memória.
    """
    print("  Aplicando One-Hot Encoding...")
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore", dtype=np.float32)
    X = encoder.fit_transform(subset)
    print(f"  Shape codificada: {X.shape}  (sparsa, {X.nnz} valores não-zero)")
    return X, encoder


def run_kmeans(X, k: int, random_state: int) -> tuple[np.ndarray, float]:
    """
    Executa MiniBatchKMeans para um dado k.

    MiniBatchKMeans é equivalente ao KMeans padrão mas usa mini-lotes
    aleatórios a cada iteração — muito mais rápido em datasets grandes,
    com pequena perda de qualidade.

    Retorna:
        labels  : array de rótulos de cluster por amostra
        inertia : soma das distâncias quadráticas ao centróide (↓ melhor)
    """
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,          # número de inicializações para evitar mínimos locais
        batch_size=4096,    # tamanho do mini-lote
        max_iter=300,
    )
    labels = model.fit_predict(X)
    return labels, model.inertia_


def compute_metrics(X, labels: np.ndarray, k: int, silhouette_sample: int) -> dict:
    """
    Calcula métricas de avaliação de clustering:

    - Silhouette Score: coesão e separação; varia de -1 a 1 (↑ melhor).
      Calculado numa amostra aleatória pois é O(n²).
    - Davies-Bouldin Index: razão dispersão intra / separação inter (↓ melhor).
    - Calinski-Harabasz Index: razão dispersão inter / intra (↑ melhor).
    - Inércia: soma das distâncias quadráticas ao centróide (↓ melhor).

    O silhouette é calculado sobre uma amostra para viabilizar o cálculo
    em datasets grandes — com n=10k já é representativo.
    """
    n = X.shape[0]
    sample_size = min(silhouette_sample, n)

    # Amostra aleatória para o silhouette
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    # Converte para denso apenas a amostra (evita materializar o dataset inteiro)
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


def print_metrics(metrics: dict, inertia: float, prev_silhouette: float | None) -> None:
    """Formata e imprime as métricas de uma iteração."""
    k = metrics["k"]
    sil = metrics["silhouette"]
    db = metrics["davies_bouldin"]
    ch = metrics["calinski_harabasz"]

    delta = ""
    if prev_silhouette is not None:
        diff = sil - prev_silhouette
        sinal = "+" if diff >= 0 else ""
        delta = f"  (Δsilhouette = {sinal}{diff:.4f})"

    print(f"\n  k={k}")
    print(f"    Silhouette Score    : {sil:.4f}{delta}")
    print(f"    Davies-Bouldin Index: {db:.4f}  (↓ melhor)")
    print(f"    Calinski-Harabasz  : {ch:.4f}  (↑ melhor)")
    print(f"    Inércia (KMeans)   : {inertia:.2f}  (↓ melhor)")


def should_stop(current_sil: float, prev_sil: float, threshold: float) -> bool:
    """
    Critério de parada: retorna True se a melhora no silhouette
    for menor que o limiar definido.
    """
    return (current_sil - prev_sil) < threshold


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    df: pd.DataFrame,
    columns: list[str] = COLUNAS_CAT_QST,
    k_start: int = 2,
    k_max: int = K_MAX,
    threshold: float = SILHOUETTE_THRESHOLD,
    random_state: int = RANDOM_STATE,
    silhouette_sample: int = SILHOUETTE_SAMPLE,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Pipeline iterativo de clustering com One-Hot + MiniBatchKMeans.

    Etapas:
        1. Seleciona e codifica as colunas com One-Hot Encoding.
        2. A partir de k=k_start, roda MiniBatchKMeans e calcula métricas.
        3. Para quando a melhora do silhouette < threshold ou k > k_max.
        4. Retorna os rótulos do melhor k e o histórico de métricas.

    Parâmetros:
        df                : DataFrame original com os dados do ENEM
        columns           : colunas categóricas a usar
        k_start           : k inicial (padrão: 2)
        k_max             : teto máximo de k
        threshold         : melhora mínima de silhouette para continuar
        random_state      : semente de aleatoriedade
        silhouette_sample : tamanho da amostra para calcular o silhouette

    Retorna:
        final_labels : pd.Series com rótulo de cluster para cada linha do df
        history_df   : pd.DataFrame com histórico de métricas por k
    """
    print("=" * 60)
    print("PIPELINE: Clustering ENEM (One-Hot + MiniBatchKMeans)")
    print("=" * 60)

    # --- Etapa 1: preparar e codificar os dados ---
    print(f"\n[1] Selecionando {len(columns)} colunas e aplicando One-Hot Encoding...")
    subset = df[columns].astype(str).copy()
    print(f"    Shape original: {subset.shape}")

    X, encoder = encode_features(subset)

    # --- Etapa 2: loop iterativo de clustering ---
    print("\n[2] Iniciando clustering iterativo...")
    print(f"    k inicial: {k_start} | limiar de parada (Δsilhouette): {threshold}")

    history = []
    best_labels = None
    best_k = k_start
    prev_silhouette = None

    for k in range(k_start, k_max + 1):
        print(f"\n{'─' * 40}")
        print(f"  Tentando k = {k}...")

        # Executa MiniBatchKMeans
        labels, inertia = run_kmeans(X, k, random_state)

        # Calcula métricas sobre amostra
        metrics = compute_metrics(X, labels, k, silhouette_sample)
        metrics["inertia"] = inertia
        metrics["labels"] = labels

        print_metrics(metrics, inertia, prev_silhouette)
        history.append(metrics)

        current_silhouette = metrics["silhouette"]

        # Atualiza o melhor k encontrado até agora
        if best_labels is None or current_silhouette > history[best_k - k_start]["silhouette"]:
            best_labels = labels.copy()
            best_k = k

        # Critério de parada
        if prev_silhouette is not None and should_stop(current_silhouette, prev_silhouette, threshold):
            print(f"\n  ⚠  Parada: Δsilhouette={current_silhouette - prev_silhouette:.4f} < {threshold}.")
            print(f"     Usando k={best_k} como resultado final.")
            break

        prev_silhouette = current_silhouette

    else:
        print(f"\n  ⚠  k_max ({k_max}) atingido. Usando k={best_k} como resultado final.")

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