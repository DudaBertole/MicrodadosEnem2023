"""
Pipeline final: MCA (n=2) + MiniBatchKMeans nos dados do ENEM.
Salva os clusters e gera análises em imagens.

Dependências:
    pip install prince scikit-learn pandas numpy matplotlib seaborn

Uso:
    python3 enem_mca_final.py
"""

import warnings
import numpy as np
import pandas as pd
import prince
import matplotlib
matplotlib.use("Agg")  # backend sem janela — necessário em terminal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

COLUNAS_CAT_QST = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025"
]

COLUNAS_NOTA = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]

N_COMPONENTS     = 2
MCA_SAMPLE_SIZE  = 50_000
MCA_BATCH_SIZE   = 100_000
K                = 2           # melhor k encontrado
RANDOM_STATE     = 42
SILHOUETTE_SAMPLE = 10_000

# Diretório de saída
OUT_DIR = Path("cluster_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")
CORES = sns.color_palette("Set2", K)

# ---------------------------------------------------------------------------
# Etapa 1: MCA
# ---------------------------------------------------------------------------

def fit_mca(cat_df: pd.DataFrame) -> prince.MCA:
    sample = cat_df.sample(n=min(MCA_SAMPLE_SIZE, len(cat_df)), random_state=RANDOM_STATE)
    mca = prince.MCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE, engine="sklearn")
    mca.fit(sample)
    # Após mca.fit(sample)
    print(mca.column_contributions_)
    return mca


def transform_batched(mca: prince.MCA, cat_df: pd.DataFrame) -> np.ndarray:
    chunks = []
    total = len(cat_df)
    for start in range(0, total, MCA_BATCH_SIZE):
        batch = cat_df.iloc[start:start + MCA_BATCH_SIZE]
        chunks.append(mca.transform(batch).values)
        print(f"    Transform: {min(start + MCA_BATCH_SIZE, total):,} / {total:,}", end="\r")
    print()
    return np.vstack(chunks)


# ---------------------------------------------------------------------------
# Etapa 2: KMeans
# ---------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
        batch_size=4096,
        max_iter=300,
    )
    return model.fit_predict(X)


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    X_s, l_s = X[idx], labels[idx]
    sil = silhouette_score(X_s, l_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = davies_bouldin_score(X_s, l_s)
        ch = calinski_harabasz_score(X_s, l_s)
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


# ---------------------------------------------------------------------------
# Etapa 3: Análises
# ---------------------------------------------------------------------------

def plot_mca_space(X: np.ndarray, labels: np.ndarray) -> None:
    """
    Análise 1 — Clusters no espaço dos 2 componentes MCA.
    Mostra a separação geométrica entre os grupos.
    """
    print("  Gerando: espaço MCA...")
    fig, ax = plt.subplots(figsize=(10, 7))

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(20_000, len(X)), replace=False)

    for k in range(K):
        mask = labels[idx] == k
        ax.scatter(
            X[idx][mask, 0], X[idx][mask, 1],
            c=[CORES[k]], label=f"Cluster {k}",
            alpha=0.3, s=5, rasterized=True,
        )

    ax.set_xlabel("Componente MCA 1", fontsize=12)
    ax.set_ylabel("Componente MCA 2", fontsize=12)
    ax.set_title("Clusters no espaço MCA (amostra de 20k pontos)", fontsize=14)
    ax.legend(markerscale=3, fontsize=11)
    plt.tight_layout()
    path = OUT_DIR / "1_espaco_mca.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Salvo: {path}")


def plot_notas_por_cluster(df: pd.DataFrame) -> None:
    """
    Análise 2 — Distribuição das notas por cluster (validação externa).
    As notas NÃO foram usadas para formar os clusters — se diferirem,
    é evidência de que os clusters capturam algo real.
    """
    print("  Gerando: distribuição das notas...")
    fig, axes = plt.subplots(1, len(COLUNAS_NOTA), figsize=(18, 5), sharey=False)

    labels_map = {
        "NU_NOTA_CN": "Ciências\nNatureza",
        "NU_NOTA_CH": "Ciências\nHumanas",
        "NU_NOTA_LC": "Ling. e\nCódigos",
        "NU_NOTA_MT": "Matemática",
        "NU_NOTA_REDACAO": "Redação",
    }

    for ax, col in zip(axes, COLUNAS_NOTA):
        data_plot = [
            df.loc[df["cluster"] == k, col].dropna().sample(
                n=min(5000, (df["cluster"] == k).sum()), random_state=RANDOM_STATE
            )
            for k in range(K)
        ]
        bp = ax.boxplot(
            data_plot,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, cor in zip(bp["boxes"], CORES):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)

        ax.set_title(labels_map[col], fontsize=11)
        ax.set_xticklabels([f"Cluster {k}" for k in range(K)], fontsize=10)
        ax.set_ylabel("Nota" if col == COLUNAS_NOTA[0] else "")

    fig.suptitle("Distribuição das notas por cluster (validação externa)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "2_notas_por_cluster.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Salvo: {path}")


def plot_perfil_socioeconomico(df: pd.DataFrame) -> None:
    """
    Análise 3 — Moda de Q006 (renda) e Q001/Q002 (escolaridade dos pais)
    por cluster. Mostra o perfil socioeconômico de cada grupo.
    """
    print("  Gerando: perfil socioeconômico...")

    # Distribuição de Q006 (faixa de renda familiar) por cluster
    q006_dist = (
        df.groupby(["cluster", "Q006"])
        .size()
        .reset_index(name="count")
    )
    q006_dist["pct"] = q006_dist.groupby("cluster")["count"].transform(lambda x: x / x.sum() * 100)

    fig, axes = plt.subplots(1, K, figsize=(14, 5), sharey=False)
    for k, ax in enumerate(axes):
        data = q006_dist[q006_dist["cluster"] == k].sort_values("Q006")
        ax.bar(data["Q006"].astype(str), data["pct"], color=CORES[k], alpha=0.8)
        ax.set_title(f"Cluster {k}", fontsize=12)
        ax.set_xlabel("Faixa de renda (Q006)", fontsize=10)
        ax.set_ylabel("% de candidatos" if k == 0 else "")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Distribuição de renda familiar (Q006) por cluster", fontsize=14)
    plt.tight_layout()
    path = OUT_DIR / "3_renda_por_cluster.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Salvo: {path}")


def plot_distribuicao_uf(df: pd.DataFrame) -> None:
    """
    Análise 4 — Proporção de cada cluster por UF.
    Mostra se há concentração geográfica dos perfis.
    """
    print("  Gerando: distribuição por UF...")

    uf_cluster = (
        df.groupby(["SG_UF_PROVA", "cluster"])
        .size()
        .unstack(fill_value=0)
    )
    # Normaliza por UF para ver proporção
    uf_pct = uf_cluster.div(uf_cluster.sum(axis=1), axis=0) * 100
    uf_pct = uf_pct.sort_values(0)  # ordena pela proporção do cluster 0

    fig, ax = plt.subplots(figsize=(14, 8))
    uf_pct.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=CORES[:K],
        alpha=0.85,
        width=0.8,
    )
    ax.set_xlabel("% de candidatos", fontsize=12)
    ax.set_ylabel("UF", fontsize=12)
    ax.set_title("Proporção dos clusters por UF", fontsize=14)
    ax.legend([f"Cluster {k}" for k in range(K)], fontsize=11)
    ax.axvline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    path = OUT_DIR / "4_distribuicao_uf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Salvo: {path}")


def salvar_resultados(df: pd.DataFrame, metrics: dict) -> None:
    """Salva o DataFrame com cluster e um resumo das métricas em CSV."""
    # Dataset completo com cluster
    path_csv = OUT_DIR / "df_com_clusters.csv"
    df.to_csv(path_csv, index=False)
    print(f"    Dataset salvo: {path_csv}")

    # Resumo das métricas
    resumo = pd.DataFrame([{
        "metodo": "MCA (n=2) + MiniBatchKMeans",
        "k": K,
        **metrics,
    }])
    path_metrics = OUT_DIR / "metricas_clustering.csv"
    resumo.to_csv(path_metrics, index=False)
    print(f"    Métricas salvas: {path_metrics}")

    # Moda por cluster
    moda_df = df.groupby("cluster")[COLUNAS_CAT_QST].agg(lambda x: x.mode()[0])
    path_moda = OUT_DIR / "moda_por_cluster.csv"
    moda_df.to_csv(path_moda)
    print(f"    Modas salvas: {path_moda}")

    # Média das notas por cluster
    media_notas = df.groupby("cluster")[COLUNAS_NOTA].mean().round(2)
    path_notas = OUT_DIR / "media_notas_por_cluster.csv"
    media_notas.to_csv(path_notas)
    print(f"    Média das notas salva: {path_notas}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("PIPELINE FINAL: MCA (n=2) + KMeans (k=2)")
    print("=" * 60)

    # --- MCA ---
    print("\n[1] Aplicando MCA...")
    cat_df = df[COLUNAS_CAT_QST].astype(str)
    mca = fit_mca(cat_df)
    print(f"  Transformando dataset em batches de {MCA_BATCH_SIZE:,}...")
    X = transform_batched(mca, cat_df)
    print(f"  Shape final: {X.shape}")

    # --- KMeans ---
    print(f"\n[2] Rodando MiniBatchKMeans (k={K})...")
    labels = run_kmeans(X, K)
    df["cluster"] = labels

    # --- Métricas ---
    print("\n[3] Calculando métricas...")
    metrics = compute_metrics(X, labels)
    print(f"  Silhouette Score    : {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz  : {metrics['calinski_harabasz']:.4f}")

    dist_str = df["cluster"].value_counts().sort_index().to_string()
    print(f"\n  Distribuição:\n{dist_str}")

    # --- Salvar dados ---
    print(f"\n[4] Salvando resultados em {OUT_DIR}...")
    salvar_resultados(df, metrics)

    # --- Análises visuais ---
    print("\n[5] Gerando análises visuais...")
    plot_mca_space(X, labels)
    plot_notas_por_cluster(df)
    plot_perfil_socioeconomico(df)
    plot_distribuicao_uf(df)

    print("\n" + "=" * 60)
    print("Concluído! Arquivos salvos em:", OUT_DIR.resolve())
    print("=" * 60)


# ---------------------------------------------------------------------------
# Bloco de execução
# ---------------------------------------------------------------------------

df = pd.read_csv("../data/df_limpo.csv", low_memory=False)

if __name__ == "__main__":
    run_pipeline(df)