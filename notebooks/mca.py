"""
Busca sistemática do número ideal de componentes MCA para clustering.

Para cada n_components testado:
    - Fita MCA em amostra, transforma dataset completo em batches
    - Roda MiniBatchKMeans com patience
    - Registra silhouette, DB, CH, variância explicada e melhor k

Dependências:
    pip install prince scikit-learn pandas numpy

Uso:
    python3 enem_mca_search.py
"""

import warnings
import numpy as np
import pandas as pd
import prince
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

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

# Valores de n_components a testar
N_COMPONENTS_RANGE = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

PATIENCE             = 3
SILHOUETTE_THRESHOLD = 0.01
K_MAX                = 15
RANDOM_STATE         = 42
SILHOUETTE_SAMPLE    = 10_000
MCA_SAMPLE_SIZE      = 50_000
MCA_BATCH_SIZE       = 100_000


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def fit_mca(cat_df: pd.DataFrame, n_components: int) -> prince.MCA:
    """Fita o MCA em uma amostra aleatória."""
    sample = cat_df.sample(n=min(MCA_SAMPLE_SIZE, len(cat_df)), random_state=RANDOM_STATE)
    mca = prince.MCA(n_components=n_components, random_state=RANDOM_STATE, engine="sklearn")
    mca.fit(sample)
    return mca


def transform_batched(mca: prince.MCA, cat_df: pd.DataFrame) -> np.ndarray:
    """Transforma o dataset completo em batches para evitar estouro de memória."""
    chunks = []
    for start in range(0, len(cat_df), MCA_BATCH_SIZE):
        batch = cat_df.iloc[start:start + MCA_BATCH_SIZE]
        chunks.append(mca.transform(batch).values)
        print(f"    Transform: {min(start + MCA_BATCH_SIZE, len(cat_df)):,} / {len(cat_df):,}", end="\r")
    print()
    return np.vstack(chunks)


def build_feature_matrix(df: pd.DataFrame, n_components: int) -> tuple[np.ndarray, float]:
    """
    Constrói a matriz de features: componentes MCA + média das notas.

    Retorna X e a variância total explicada pelo MCA.
    """
    cat_df = df[COLUNAS_CAT].astype(str)

    print(f"  Fitting MCA (n_components={n_components})...")
    mca = fit_mca(cat_df, n_components)

    # Variância explicada pelos componentes
    explained = sum(mca.eigenvalues_summary["% of variance"].str.rstrip("%").astype(float)) / 100


    print(f"  Variância explicada: {explained:.1%}")
    print(f"  Transformando dataset completo...")
    X_mca = transform_batched(mca, cat_df)

    # Média das notas normalizada
    media_notas = df[COLUNAS_NOTA].mean(axis=1).values.reshape(-1, 1)
    scaler = StandardScaler()
    media_notas_scaled = scaler.fit_transform(media_notas)

    X = np.hstack([X_mca, media_notas_scaled])
    return X, float(explained)


def run_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Executa MiniBatchKMeans para um dado k."""
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
        batch_size=4096,
        max_iter=300,
    )
    labels = model.fit_predict(X)
    return labels, model.inertia_


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Calcula métricas sobre amostra aleatória de SILHOUETTE_SAMPLE pontos."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    X_s, l_s = X[idx], labels[idx]

    sil = silhouette_score(X_s, l_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = davies_bouldin_score(X_s, l_s)
        ch = calinski_harabasz_score(X_s, l_s)

    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


def find_best_k(X: np.ndarray) -> tuple[int, float, dict]:
    """
    Loop iterativo com patience para encontrar o melhor k dado um X.

    Retorna o melhor k, melhor silhouette e as métricas correspondentes.
    """
    best_labels = None
    best_k      = 2
    best_sil    = -np.inf
    no_improve  = 0
    best_metrics = {}

    for k in range(2, K_MAX + 1):
        labels, _ = run_kmeans(X, k)
        metrics   = compute_metrics(X, labels)
        sil       = metrics["silhouette"]

        if sil > best_sil + SILHOUETTE_THRESHOLD:
            best_sil     = sil
            best_labels  = labels
            best_k       = k
            no_improve   = 0
            best_metrics = metrics
            print(f"    k={k}  silhouette={sil:.4f}  ✓ novo melhor")
        else:
            no_improve += 1
            print(f"    k={k}  silhouette={sil:.4f}  → sem melhora ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            break

    return best_k, best_sil, best_metrics


# ---------------------------------------------------------------------------
# Busca principal
# ---------------------------------------------------------------------------

def run_search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Testa cada valor de n_components em N_COMPONENTS_RANGE.

    Para cada n:
        1. Fita MCA e transforma o dataset
        2. Roda o loop de clustering com patience
        3. Registra variância explicada, melhor k e métricas

    Retorna um DataFrame com o resumo de todos os testes,
    ordenado por silhouette decrescente.
    """
    print("=" * 60)
    print("BUSCA: n_components MCA x silhouette")
    print("=" * 60)

    results = []

    for n in N_COMPONENTS_RANGE:
        print(f"\n{'━' * 60}")
        print(f"  Testando n_components = {n}")
        print(f"{'━' * 60}")

        # Constrói o espaço de features com n componentes
        X, explained = build_feature_matrix(df, n)

        # Encontra o melhor k para esse espaço
        print(f"  Buscando melhor k (patience={PATIENCE}, K_MAX={K_MAX})...")
        best_k, best_sil, best_metrics = find_best_k(X)

        result = {
            "n_components"      : n,
            "variancia_explicada": f"{explained:.1%}",
            "melhor_k"          : best_k,
            "silhouette"        : round(best_sil, 4),
            "davies_bouldin"    : round(best_metrics.get("davies_bouldin", np.nan), 4),
            "calinski_harabasz" : round(best_metrics.get("calinski_harabasz", np.nan), 2),
        }
        results.append(result)

        print(f"\n  Resultado: n={n} | k={best_k} | silhouette={best_sil:.4f} "
              f"| variância={explained:.1%}")

    # Resumo final
    results_df = pd.DataFrame(results).sort_values("silhouette", ascending=False)

    print("\n" + "=" * 60)
    print("RESUMO — ordenado por silhouette (↓)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    best = results_df.iloc[0]
    print(f"\n✓ Melhor configuração:")
    print(f"  n_components = {best['n_components']}")
    print(f"  k            = {best['melhor_k']}")
    print(f"  silhouette   = {best['silhouette']}")
    print(f"  variância    = {best['variancia_explicada']}")

    return results_df


# ---------------------------------------------------------------------------
# Bloco de execução
# ---------------------------------------------------------------------------

df = pd.read_csv("../data/df_limpo.csv", low_memory=False)

if __name__ == "__main__":
    search_results = run_search(df)
    search_results.to_csv("../data/mca_search_results.csv", index=False)
    print("\n[INFO] Resultados salvos em ../data/mca_search_results.csv")