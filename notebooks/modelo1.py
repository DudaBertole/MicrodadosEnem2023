"""
Clustering iterativo do questionário socioeconômico do ENEM
usando distância de Gower + K-Medoids (FasterPAM).

Dependências:
    pip install gower kmedoids scikit-learn pandas numpy

Uso:
    Certifique-se de que `df` está carregado no escopo antes de importar/rodar.
    O script pode ser executado como módulo ou colado num notebook.
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

# Colunas do questionário socioeconômico (apenas categoricas)
COLUNAS_CAT_QST = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025"
]

# Limiar mínimo de melhora do coeficiente de silhueta para continuar iterando.
# Se a melhora absoluta for menor que isso, o algoritmo para.
SILHOUETTE_THRESHOLD = 0.01

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
    simples (0 = mesma categoria, 1 = categorias diferentes), e variáveis
    numéricas com distância normalizada pelo range — aqui todas são categóricas,
    então a função vai usar apenas a parcela categórica.

    Retorna uma matriz quadrada (n x n) de float32 com valores em [0, 1].
    """
    print("  Calculando matriz de distância de Gower...")
    # gower.gower_matrix retorna np.ndarray de shape (n, n)
    dist_matrix = gower.gower_matrix(data)
    print(f"  Matriz calculada: {dist_matrix.shape}, dtype={dist_matrix.dtype}")
    return dist_matrix.astype(np.float64)  # kmedoids prefere float64


def run_kmedoids(dist_matrix: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """
    Executa o FasterPAM (variante eficiente do PAM) sobre a matriz de
    distância pré-computada para um dado k.

    Retorna:
        labels      : array de rótulos de cluster (int) por amostra
        inertia     : soma das distâncias intra-cluster (loss do FasterPAM)
    """
    result = kmedoids.fasterpam(dist_matrix, k, random_state=RANDOM_STATE)
    labels = np.array(result.labels)
    return labels, result.loss


def compute_metrics(dist_matrix: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """
    Calcula as métricas de avaliação de clustering:

    - Silhouette Score: mede coesão e separação; varia de -1 a 1,
      quanto maior melhor.
    - Davies-Bouldin Index: razão média entre dispersão intra-cluster e
      separação entre clusters; quanto menor melhor.
    - Calinski-Harabasz Index: razão entre dispersão inter e intra-cluster;
      quanto maior melhor. Calculado sobre a matriz de distâncias como
      representação numérica.
    - Inércia (loss FasterPAM): soma das distâncias de cada ponto ao seu
      medoide; quanto menor melhor.

    Nota: silhouette_score com metric='precomputed' aceita a matriz de
    distâncias diretamente, sem precisar das features originais.
    """
    silhouette = silhouette_score(dist_matrix, labels, metric="precomputed")

    # Davies-Bouldin e Calinski-Harabasz não aceitam matriz pré-computada
    # diretamente; usamos a própria matriz como "espaço de features" —
    # é uma aproximação válida para fins comparativos entre valores de k.
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
    print(f"    Inércia (FasterPAM): {inertia:.4f}  (↓ melhor)")


def should_stop(current_sil: float, prev_sil: float, threshold: float) -> bool:
    """
    Critério de parada: retorna True se a melhora no coeficiente de
    silhueta não for significativa (menor que o limiar definido).
    """
    improvement = current_sil - prev_sil
    return improvement < threshold


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    df: pd.DataFrame,
    columns: list[str] = COLUNAS_CAT_QST,
    k_start: int = 2,
    k_max: int = K_MAX,
    threshold: float = SILHOUETTE_THRESHOLD,
    sample_size: int = 30_000,       # tamanho da amostra para treino
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.DataFrame]:

    print("=" * 60)
    print("PIPELINE: Clustering socioeconômico ENEM (Gower + K-Medoids)")
    print("=" * 60)

    # --- Etapa 1: preparar os dados ---
    subset = df[columns].copy()
    for col in subset.columns:
        if subset[col].dtype != object:
            subset[col] = subset[col].astype(str)

    # --- Amostragem estratificada por Q006 (renda), se existir ---
    # Garante representatividade socioeconômica na amostra de treino
    n = len(subset)
    if n > sample_size:
        strat_col = "Q006" if "Q006" in subset.columns else subset.columns[0]
        print(f"\n[!] Dataset grande ({n:,} linhas). Amostrando {sample_size:,} "
              f"linhas (estratificado por {strat_col})...")
        sample_idx = (
            subset
            .groupby(strat_col, group_keys=False)
            .apply(lambda g: g.sample(
                n=max(1, int(sample_size * len(g) / n)),
                random_state=random_state
            ))
            .index
        )
        # Ajuste fino caso o arredondamento dê mais/menos que sample_size
        sample_idx = sample_idx[:sample_size]
        sample_df = subset.loc[sample_idx]
    else:
        sample_df = subset
        sample_idx = subset.index

    print(f"    Amostra de treino: {len(sample_df):,} linhas")
    print(f"    Memória estimada da matriz Gower: "
          f"{len(sample_df)**2 * 4 / 1e9:.2f} GB")

    # --- Etapa 2: matriz de Gower só na amostra ---
    print("\n[2] Calculando distância de Gower (amostra)...")
    dist_matrix = compute_gower_matrix(sample_df)

    # --- Etapa 3: loop iterativo de clustering na amostra ---
    print("\n[3] Iniciando clustering iterativo...")
    history = []
    best_labels_sample = None
    best_k = k_start
    prev_silhouette = None

    for k in range(k_start, k_max + 1):
        print(f"\n{'─' * 40}")
        print(f"  Tentando k = {k}...")

        labels, inertia = run_kmedoids(dist_matrix, k)
        metrics = compute_metrics(dist_matrix, labels, k)
        metrics["inertia"] = inertia
        metrics["labels"] = labels
        print_metrics(metrics, inertia, prev_silhouette)
        history.append(metrics)

        current_silhouette = metrics["silhouette"]

        if best_labels_sample is None or current_silhouette > history[best_k - k_start]["silhouette"]:
            best_labels_sample = labels
            best_k = k

        if prev_silhouette is not None and should_stop(current_silhouette, prev_silhouette, threshold):
            print(f"\n  ⚠  Parada: Δsilhouette={current_silhouette - prev_silhouette:.4f} < {threshold}. "
                  f"Usando k={best_k}.")
            break

        prev_silhouette = current_silhouette

    # --- Etapa 4: projetar o resto do dataset nos clusters encontrados ---
    # Para cada ponto fora da amostra, calcula distância de Gower até
    # os medoides e atribui o cluster do medoide mais próximo.
    best_row = next(r for r in history if r["k"] == best_k)
    final_labels_sample = pd.Series(best_row["labels"], index=sample_idx, name="cluster")

    if n > sample_size:
        print(f"\n[4] Projetando {n - len(sample_idx):,} pontos restantes nos medoides...")

        # Recupera os índices dos medoides dentro da amostra
        km_result = kmedoids.fasterpam(dist_matrix, best_k, random_state=random_state)
        medoid_positions = km_result.medoids           # posições na amostra
        medoid_idx = sample_df.iloc[medoid_positions].index
        medoids_df = subset.loc[medoid_idx]            # features dos medoides

        # Pontos que não estão na amostra
        out_idx = subset.index.difference(sample_idx)
        out_df = subset.loc[out_idx]

        # Distância de Gower de cada ponto externo até cada medoide
        # gower_matrix(X, Y) → shape (|X|, |Y|)
        dist_to_medoids = gower.gower_matrix(out_df, medoids_df)  # (m, k)
        assigned_clusters = dist_to_medoids.argmin(axis=1)
        final_labels_out = pd.Series(assigned_clusters, index=out_idx, name="cluster")

        # Une amostra + resto e reordena igual ao df original
        final_labels = pd.concat([final_labels_sample, final_labels_out]).reindex(df.index)
    else:
        final_labels = final_labels_sample.reindex(df.index)

    # --- Resumo final ---
    print("\n" + "=" * 60)
    print(f"RESULTADO FINAL: k = {best_k}")
    print(f"  Silhouette : {best_row['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {best_row['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz: {best_row['calinski_harabasz']:.4f}")
    print(f"\nDistribuição dos clusters (dataset completo):")
    print(final_labels.value_counts().sort_index().to_string())
    print("=" * 60)

    history_df = pd.DataFrame([
        {k_: v for k_, v in row.items() if k_ != "labels"}
        for row in history
    ])
    return final_labels, history_df


# ---------------------------------------------------------------------------
# Bloco de execução (quando rodado como script standalone)
# ---------------------------------------------------------------------------

df= pd.read_csv("../data/df_limpo.csv")

if __name__ == "__main__":
    # Aqui assumimos que `df` já existe no namespace.
    # Em um notebook, basta chamar run_clustering_pipeline(df) diretamente.
    try:
        cluster_labels, metrics_history = run_clustering_pipeline(df, sample_size=15_000)

        # Adiciona os rótulos ao DataFrame original
        df["cluster"] = cluster_labels

        print("\n[INFO] Coluna 'cluster' adicionada ao DataFrame `df`.")
        print("[INFO] Histórico de métricas disponível em `metrics_history`.")

    except NameError:
        print("[ERRO] A variável `df` não está definida no escopo.")
        print("       Carregue o dataset antes de executar este script.")