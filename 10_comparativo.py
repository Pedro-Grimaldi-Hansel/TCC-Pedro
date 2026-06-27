import pickle
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from config import ESTADOS, DIR_RESULTADOS

ALGOS = ["KMeans", "DBSCAN", "HDBSCAN", "Aglomerativo"]
MIN_PONTOS = 50   # par só é comparado se sobrarem ao menos tantos pontos não-ruído

# Lê o pickle que as etapas 05–08 foram acumulando.
with open(DIR_RESULTADOS + "resultados.pkl", "rb") as f:
    resultados = pickle.load(f)

pares = list(combinations(ALGOS, 2))
colunas = [f"{a}_vs_{b}" for a, b in pares]

# ── ARI de cada par de algoritmos, por estado ─────────────────────────────────
rows = {}
incompativeis = []  # (estado, col, n_a, n_b) — pares pulados por tamanho diferente
for estado in ESTADOS:
    if estado not in resultados:
        continue
    res_estado = resultados[estado]
    row = {}
    for algo_a, algo_b in pares:
        col = f"{algo_a}_vs_{algo_b}"
        if algo_a not in res_estado or algo_b not in res_estado:
            row[col] = np.nan
            continue
        labels_a = np.asarray(res_estado[algo_a]["labels"])
        labels_b = np.asarray(res_estado[algo_b]["labels"])
        # O ARI compara ponto a ponto. Algoritmos que rodam sobre subamostra (ex.:
        # o Aglomerativo) têm menos rótulos e não alinham com os que cobrem todos
        # os pontos — esse par é pulado.
        if labels_a.shape != labels_b.shape:
            incompativeis.append((estado, col, labels_a.shape[0], labels_b.shape[0]))
            row[col] = np.nan
            continue
        # Compara só onde os dois atribuíram cluster (descarta o ruído de ambos).
        mask = (labels_a >= 0) & (labels_b >= 0)
        if mask.sum() < MIN_PONTOS:
            row[col] = np.nan
        else:
            row[col] = adjusted_rand_score(labels_a[mask], labels_b[mask])
    rows[estado] = row

df_ari = pd.DataFrame(rows).T.reindex(columns=colunas)
df_ari.index.name = "Estado"

print("\n=== ARI entre algoritmos por estado ===")
print(df_ari.round(3).to_string())

df_ari.to_csv(DIR_RESULTADOS + "ari_entre_algoritmos.csv")

# ── Resumo por par: média e em quais estados há concordância moderada (>0.3) ──
print("\n=== Resumo por par ===")
for col in colunas:
    media = df_ari[col].mean()
    moderados = df_ari.index[df_ari[col] > 0.3].tolist()
    print(f"{col}: média={media:.3f} | ARI>0.3: {moderados if moderados else 'nenhum'}")

# ── Lista os pares que não deram para comparar (tamanhos incompatíveis) ───────
if incompativeis:
    print("\n=== Pares PULADOS (tamanhos incompatíveis — sem ARI) ===")
    print("  Causa: um dos algoritmos rodou sobre subamostra (ex.: Aglomerativo),")
    print("  então não há alinhamento ponto a ponto com os que cobrem todos os pontos.")
    afetados = sorted({col for _, col, _, _ in incompativeis})
    print(f"  Pares afetados: {', '.join(afetados)}")
    for estado, col, n_a, n_b in incompativeis:
        print(f"    {estado:<4} {col}: {n_a:,} vs {n_b:,}")
