import prince
import pandas as pd

COLUNAS_CAT = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025",
    "TP_ESCOLA", "TP_COR_RACA", "TP_FAIXA_ETARIA",
    "TP_ST_CONCLUSAO", "SG_UF_PROVA",
]

df = pd.read_csv("../data/df_limpo.csv", low_memory=False)
sample = df[COLUNAS_CAT].astype(str).sample(n=50_000, random_state=42)

mca = prince.MCA(n_components=30, random_state=42, engine="sklearn")
mca.fit(sample)

print(mca.eigenvalues_summary)