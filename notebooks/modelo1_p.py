"""
Clustering iterativo do questionário socioeconômico do ENEM
usando distância de Gower + K-Medoids (FasterPAM).

Dependências:
    pip install gower kmedoids scikit-learn pandas numpy

Uso:
    python3 enem_gower_clustering.py
"""

import warnings
import numpy as np
import pandas as pd
import gower
import kmedoids
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


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def compute_gower_matrix(data: pd.DataFrame) -> np.ndarray:
    """
    Calcula a matriz de distância de Gower para um DataFrame de variáveis
    categóricas. Gower trata variáveis categóricas como correspondência
    simples (0 = mesma categoria, 1 = categorias diferentes).

    Retorna uma matriz quadrada (n x n) de float64 com valores em [0, 1].
    """
    print("  Calculando matriz de distância de Gower...")
    dist_matrix = gower.gower_matrix(data)
    print(f"  Matriz calculada: {dist_matrix.shape}, dtype={dist_matrix.dtype}")
    return dist_matrix.astype(np.float64)


def run_kmedoids(dist_matrix: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """
    Executa o FasterPAM sobre a matriz de distância pré-computada.

    Retorna:
        labels  : array de rótulos de cluster por amostra
        loss    : soma das distâncias intra-cluster (↓ melhor)
    """
    result = kmedoids.fasterpam(dist_matrix, k, random_state=RANDOM_STATE)
    return np.array(result.labels), result.loss


def compute_metrics(dist_matrix: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """
    Calcula métricas de avaliação de clustering:

    - Silhouette Score : coesão e separação; -1 a 1 (↑ melhor)
    - Davies-Bouldin   : dispersão intra / separação inter (↓ melhor)
    - Calinski-Harabasz: dispersão inter / intra (↑ melhor)

    Silhouette usa metric='precomputed' com a matriz de Gower diretamente.
    DB e CH usam a matriz como espaço de features — aproximação válida
    para comparação entre valores de k.
    """
    silhouette = silhouette_score(dist_matrix, labels, metric="precomputed")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        davies_bouldin = davies_bouldin_score(dist_matrix, labels)
        calinski_harabasz = calinski_harabasz_score(dist_matrix, labels)

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
        diff = sil - best_sil
        delta = f"  (Δbest = {diff:+.4f})"

    print(f"\n  k={metrics['k']}")
    print(f"    Silhouette Score    : {sil:.4f}{delta}")
    print(f"    Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}  (↓ melhor)")
    print(f"    Calinski-Harabasz  : {metrics['calinski_harabasz']:.4f}  (↑ melhor)")
    print(f"    Inércia (FasterPAM): {inertia:.4f}  (↓ melhor)")


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
    sample_size: int = 15_000,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Pipeline iterativo de clustering com Gower + K-Medoids.

    Critério de parada por patience: só para após `patience` iterações
    consecutivas sem melhora significativa no silhouette, evitando parar
    prematuramente em mínimos locais.

    Etapas:
        1. Amostragem estratificada por Q006 (se n > sample_size).
        2. Cálculo da matriz de Gower na amostra (feito uma única vez).
        3. Loop iterativo de K-Medoids com critério de parada por patience.
        4. Projeção dos pontos fora da amostra nos medoides encontrados.
    """
    print("=" * 60)
    print("PIPELINE: Clustering ENEM (Gower + K-Medoids)")
    print("=" * 60)
    print(f"    Patience: {patience} iterações sem melhora")

    # --- Etapa 1: preparar e amostrar os dados ---
    print(f"\n[1] Preparando dados...")
    subset = df[columns].copy()
    for col in subset.columns:
        if subset[col].dtype != object:
            subset[col] = subset[col].astype(str)

    n = len(subset)
    if n > sample_size:
        strat_col = "Q006" if "Q006" in subset.columns else subset.columns[0]
        print(f"    Dataset grande ({n:,} linhas). Amostrando {sample_size:,} "
              f"linhas (estratificado por {strat_col})...")
        sample_idx = (
            subset
            .groupby(strat_col, group_keys=False)
            .apply(lambda g: g.sample(
                n=max(1, int(sample_size * len(g) / n)),
                random_state=random_state,
            ), include_groups=False)
            .sample(frac=1, random_state=random_state)  # shuffle global
            .index
        )
        sample_idx = sample_idx[:sample_size]
        sample_df = subset.loc[sample_idx]
    else:
        sample_df = subset
        sample_idx = subset.index

    print(f"    Amostra de treino: {len(sample_df):,} linhas")
    print(f"    Memória estimada da matriz Gower: {len(sample_df)**2 * 4 / 1e9:.2f} GB")

    # --- Etapa 2: matriz de Gower na amostra ---
    print("\n[2] Calculando distância de Gower (amostra)...")
    dist_matrix = compute_gower_matrix(sample_df)

    # --- Etapa 3: loop iterativo com patience ---
    print("\n[3] Iniciando clustering iterativo...")
    print(f"    k inicial: {k_start} | patience: {patience} | threshold: {threshold}")

    history = []
    best_labels_sample = None
    best_k = k_start
    best_silhouette = -np.inf
    no_improve_count = 0

    for k in range(k_start, k_max + 1):
        print(f"\n{'─' * 40}")
        print(f"  Tentando k = {k}...")

        labels, inertia = run_kmedoids(dist_matrix, k)
        metrics = compute_metrics(dist_matrix, labels, k)
        metrics["inertia"] = inertia
        metrics["labels"] = labels

        print_metrics(metrics, inertia, best_silhouette if best_labels_sample is not None else None)
        history.append(metrics)

        current_silhouette = metrics["silhouette"]

        # Verifica se houve melhora significativa
        if current_silhouette > best_silhouette + threshold:
            best_silhouette = current_silhouette
            best_labels_sample = labels.copy()
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

    # --- Etapa 4: projetar pontos restantes nos medoides ---
    best_row = next(r for r in history if r["k"] == best_k)
    final_labels_sample = pd.Series(best_row["labels"], index=sample_idx, name="cluster")

    if n > sample_size:
        print(f"\n[4] Projetando {n - len(sample_idx):,} pontos restantes nos medoides...")

        # Reexecuta o FasterPAM para recuperar os índices dos medoides
        km_result = kmedoids.fasterpam(dist_matrix, best_k, random_state=random_state)
        medoid_idx = sample_df.iloc[km_result.medoids].index
        medoids_df = subset.loc[medoid_idx]

        out_idx = subset.index.difference(sample_idx)
        out_df = subset.loc[out_idx]

        # Processa em batches para não estourar memória
        BATCH_SIZE = 50_000
        all_assigned = []
        for start in range(0, len(out_df), BATCH_SIZE):
            batch = out_df.iloc[start:start + BATCH_SIZE]
            dist_batch = gower.gower_matrix(batch, medoids_df)
            all_assigned.append(dist_batch.argmin(axis=1))

        assigned_clusters = np.concatenate(all_assigned)
        final_labels_out = pd.Series(assigned_clusters, index=out_idx, name="cluster")
        final_labels = pd.concat([final_labels_sample, final_labels_out]).reindex(df.index)
    else:
        final_labels = final_labels_sample.reindex(df.index)

    # --- Resumo final ---
    print("\n" + "=" * 60)
    print(f"RESULTADO FINAL: k = {best_k}")
    print(f"  Silhouette Score    : {best_row['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {best_row['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz  : {best_row['calinski_harabasz']:.4f}")
    print(f"  Inércia             : {best_row['inertia']:.4f}")
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
    cluster_labels, metrics_history = run_clustering_pipeline(df, sample_size=15_000)
    df["cluster"] = cluster_labels
    print("\n[INFO] Coluna 'cluster' adicionada ao DataFrame `df`.")
    print("[INFO] Histórico de métricas disponível em `metrics_history`.")