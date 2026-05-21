import time
import gower
import numpy as np
import pandas as pd

COLUNAS_CAT_QST = [
    "Q001", "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q009", "Q010",
    "Q011", "Q012", "Q013", "Q014", "Q015",
    "Q016", "Q017", "Q018", "Q019", "Q020",
    "Q021", "Q022", "Q023", "Q024", "Q025"
]

df = pd.read_csv("../data/df_limpo.csv", low_memory=False)
subset = df[COLUNAS_CAT_QST].astype(str)  # <-- aqui

sizes = [500, 1000, 2000, 5000]
times = []

for n in sizes:
    sample = subset.sample(n=n, random_state=42)
    start = time.time()
    gower.gower_matrix(sample)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  n={n:>5}: {elapsed:.2f}s")

coeffs = np.polyfit(sizes, times, deg=2)
poly = np.poly1d(coeffs)

for target in [10_000, 30_000, 50_000]:
    est = poly(target)
    print(f"  n={target:>6} (estimado): {est:.0f}s  (~{est/60:.1f} min)")