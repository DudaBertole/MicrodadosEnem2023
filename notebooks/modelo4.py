"""
Clustering iterativo do questionário socioeconômico do ENEM
usando K-Prototypes (numérica + categóricas mistas).

Dependências:
    pip install kmodes scikit-learn pandas numpy

Uso:
    python3 enem_kprototypes_clustering.py
"""

import warnings
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------

COLUNAS_CAT = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025",
    "TP_ESCOLA", "TP_COR_RACA", "TP_FAIXA_ETARIA",
    "TP_ST_CONCLUSAO", "SG_UF_PROVA",
]

COLUNAS_NOTA = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]

PATIENCE             = 3
SILHOUETTE_THRESHOLD = 0.01
K_MAX                = 8       # teto reduzido para economizar tempo/memória
RANDOM_STATE         = 42
SILHOUETTE_SAMPLE    = 10_000

# Tamanho da amostra para treino — evita derrubar o WSL
SAMPLE_SIZE          = 50_000

# n_init=1 para exploração; aumente para 3-5 só no modelo final
N_INIT               = 1


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int], StandardScaler]:
    """
    Monta a matriz de features para o K-Prototypes.

    Estrutura de X: [media_nota_scaled | Q001 | ... | Q025 | TP_ESCOLA | ...]
    K-Prototypes exige que as categóricas estejam nas últimas posições
    e recebe os índices delas via parâmetro `categorical`.

    Retorna:
        X       : array completo (numérica + categóricas)
        X_num   : array numérico com LabelEncoding (para métricas)
        cat_idx : índices das colunas categóricas em X
        scaler  : StandardScaler fitado (para transformar novos dados)
    """
    print("  Calculando média das notas...")
    media_notas = df[COLUNAS_NOTA].mean(axis=1).values.reshape(-1, 1)

    # Normaliza para ter escala comparável às categóricas
    # Sem isso a nota dominaria a distância pelo range maior
    scaler = StandardScaler()
    media_notas_scaled = scaler.fit_transform(media_notas)

    cat_df = df[COLUNAS_CAT].astype(str)

    # X: coluna numérica primeiro, categóricas depois
    X = np.hstack([media_notas_scaled, cat_df.values])

    # Índices das colunas categóricas (tudo após a coluna 0)
    cat_idx = list(range(1, X.shape[1]))

    # Versão numérica para silhouette/DB/CH
    X_num = np.zeros((len(df), 1 + len(COLUNAS_CAT)), dtype=np.float32)
    X_num[:, 0] = media_notas_scaled.ravel()
    for i, col in enumerate(COLUNAS_CAT):
        le = LabelEncoder()
        X_num[:, i + 1] = le.fit_transform(cat_df[col].values)

    print(f"  Shape final: {X.shape}  (1 numérica + {len(cat_idx)} categóricas)")
    return X, X_num, cat_idx, scaler


def sample_stratified(X: np.ndarray, X_num: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Amostragem aleatória simples para o treino.

    Retorna X_sample, X_num_sample e os índices da amostra.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(size, len(X)), replace=False)
    return X[idx], X_num[idx], idx


def run_kprototypes(X_sample: np.ndarray, cat_idx: list[int], k: int) -> tuple[KPrototypes, np.ndarray, float]:
    """
    Treina K-Prototypes na amostra.

    Retorna o modelo fitado (para predict posterior), os labels da amostra
    e o custo total.
    """
    model = KPrototypes(
        n_clusters=k,
        init="Huang",
        n_init=N_INIT,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    labels = model.fit_predict(X_sample, categorical=cat_idx)
    return model, labels, model.cost_


def predict_batched(model: KPrototypes, X: np.ndarray, cat_idx: list[int], batch_size: int = 50_000) -> np.ndarray:
    """
    Prediz os clusters para o dataset completo em batches.

    Evita materializar o dataset inteiro na memória de uma vez,
    reduzindo o pico de uso de RAM.
    """
    all_labels = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        batch_labels = model.predict(batch, categorical=cat_idx)
        all_labels.append(batch_labels)
    return np.concatenate(all_labels)


def compute_metrics(X_num: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calcula métricas sobre uma amostra aleatória.

    Silhouette é O(n²) — calculado em amostra de 10k pontos.
    DB e CH também calculados na mesma amostra para consistência.
    """
    n = X_num.shape[0]
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(n, size=min(SILHOUETTE_SAMPLE, n), replace=False)

    X_s = X_num[sample_idx]
    l_s = labels[sample_idx]

    sil = silhouette_score(X_s, l_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = davies_bouldin_score(X_s, l_s)
        ch = calinski_harabasz_score(X_s, l_s)

    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_clustering_pipeline(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Pipeline iterativo de clustering com K-Prototypes.

    Estratégia de dois estágios para economizar memória:
        1. Treina em amostra de SAMPLE_SIZE linhas (busca do melhor k)
        2. Com o melhor k encontrado, prediz o dataset completo em batches
    """
    print("=" * 60)
    print("PIPELINE: Clustering ENEM (K-Prototypes)")
    print("=" * 60)
    print(f"    Patience : {PATIENCE} | K_MAX: {K_MAX} | n_init: {N_INIT}")
    print(f"    Treino   : amostra de {SAMPLE_SIZE:,} linhas")
    print(f"    Silhouette: amostra de {SILHOUETTE_SAMPLE:,} pontos")

    # --- Etapa 1: preparar os dados ---
    print("\n[1] Preparando dados...")
    X, X_num, cat_idx, scaler = prepare_data(df)

    # --- Etapa 2: amostrar para treino ---
    print(f"\n[2] Amostrando {SAMPLE_SIZE:,} linhas para treino...")
    X_sample, X_num_sample, sample_idx = sample_stratified(X, X_num, SAMPLE_SIZE)
    print(f"    Shape da amostra: {X_sample.shape}")

    # --- Etapa 3: loop iterativo com patience ---
    print("\n[3] Iniciando clustering iterativo (treino na amostra)...")

    history    = []
    best_model = None
    best_k     = 2
    best_sil   = -np.inf
    no_improve = 0

    for k in range(2, K_MAX + 1):
        print(f"\n{'─' * 40}")
        print(f"  Tentando k = {k}...")

        # Treina na amostra
        model, labels_sample, cost = run_kprototypes(X_sample, cat_idx, k)

        # Métricas na amostra de treino
        metrics = compute_metrics(X_num_sample, labels_sample)
        metrics.update({"k": k, "cost": cost})
        history.append(metrics)

        sil = metrics["silhouette"]
        delta_str = f"  (Δbest = {sil - best_sil:+.4f})" if best_model is not None else ""
        print(f"\n  k={k}")
        print(f"    Silhouette Score    : {sil:.4f}{delta_str}")
        print(f"    Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}  (↓ melhor)")
        print(f"    Calinski-Harabasz  : {metrics['calinski_harabasz']:.4f}  (↑ melhor)")
        print(f"    Custo              : {cost:.2f}  (↓ melhor)")

        if sil > best_sil + SILHOUETTE_THRESHOLD:
            best_sil   = sil
            best_model = model
            best_k     = k
            no_improve = 0
            print(f"  ✓  Novo melhor: k={k}, silhouette={sil:.4f}")
        else:
            no_improve += 1
            print(f"  →  Sem melhora significativa ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print(f"\n  ⚠  Parada: {PATIENCE} iterações sem melhora. Melhor k={best_k}.")
            break
    else:
        print(f"\n  ⚠  k_max ({K_MAX}) atingido. Usando k={best_k}.")

    # --- Etapa 4: prediz dataset completo em batches ---
    print(f"\n[4] Predizendo dataset completo ({len(df):,} linhas) em batches...")
    final_labels_arr = predict_batched(best_model, X, cat_idx)
    final_labels = pd.Series(final_labels_arr, index=df.index, name="cluster")

    # --- Resumo final ---
    best_row = next(r for r in history if r["k"] == best_k)
    print("\n" + "=" * 60)
    print(f"RESULTADO FINAL: k = {best_k}")
    print(f"  Silhouette Score    : {best_row['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {best_row['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz  : {best_row['calinski_harabasz']:.4f}")
    print(f"  Custo              : {best_row['cost']:.2f}")
    print(f"\nDistribuição dos clusters (dataset completo):")
    print(final_labels.value_counts().sort_index().to_string())
    print("=" * 60)

    history_df = pd.DataFrame(history)
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