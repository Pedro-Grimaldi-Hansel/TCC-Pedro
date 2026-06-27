import matplotlib
matplotlib.use('Agg')

import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from config import CORES_ESTADO, DIR_DADOS, DIR_FIGURAS, ESTADOS
from funcoes import preparar_features_ohe

# ── Carrega dados ─────────────────────────────────────────────────────────────
os.makedirs(DIR_FIGURAS, exist_ok=True)
df_homicidios = pd.read_parquet(DIR_DADOS + "homicidios.parquet")


# ── Função local ──────────────────────────────────────────────────────────────

def hopkins_statistic(X: np.ndarray, sample_size: int = 150,
                      n_runs: int = 10, seed: int = 42) -> dict:
    """Estatística de Hopkins: mede o quanto os dados tendem a formar clusters.

    Compara a distância de pontos reais ao vizinho mais próximo com a de pontos
    aleatórios (uniformes na mesma caixa). Interpretação:
      H ≈ 1.0  → forte tendência a clustering
      H ≈ 0.5  → dados aleatórios (clusterizar não faria sentido)
      H ≈ 0.0  → dados num grid regular (raro)
    Repete `n_runs` vezes (amostras diferentes) e retorna média/desvio. Usa a soma
    das distâncias sem elevar a d, por estabilidade numérica em alta dimensão.
    """
    n, d = X.shape
    scores = []

    for i in range(n_runs):
        rng = np.random.default_rng(seed + i)

        # Amostra de pontos reais.
        idx      = rng.choice(n, size=sample_size, replace=False)
        X_sample = X[idx]

        # Pontos artificiais uniformes dentro da caixa [min, max] de cada dimensão.
        mins   = X.min(axis=0)
        maxs   = X.max(axis=0)
        X_rand = rng.uniform(mins, maxs, size=(sample_size, d))

        # Vizinho mais próximo medido contra o resto dos dados reais.
        mask       = np.ones(n, dtype=bool)
        mask[idx]  = False
        X_restante = X[mask]

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
        nn.fit(X_restante)

        dist_real, _ = nn.kneighbors(X_sample)
        dist_rand, _ = nn.kneighbors(X_rand)

        soma_rand = dist_rand.sum()
        soma_real = dist_real.sum()
        scores.append(soma_rand / (soma_rand + soma_real))

    media = float(np.mean(scores))
    std   = float(np.std(scores))

    if media >= 0.75:
        interpretacao = "FORTE tendencia a clustering - dados bem estruturados"
    elif media >= 0.60:
        interpretacao = "MODERADA tendencia a clustering - adequado para K-Means"
    elif media >= 0.50:
        interpretacao = "FRACA tendencia - clusters podem nao ser naturais"
    else:
        interpretacao = "Dados proximos de aleatorios - revisar abordagem"

    return {"media": media, "std": std, "valores": scores,
            "interpretacao": interpretacao}


# ── Parâmetros ────────────────────────────────────────────────────────────────
HOPKINS_SAMPLE = 150
HOPKINS_RUNS   = 10

resultados_hopkins = {}

# ── Loop por estado ───────────────────────────────────────────────────────────
for ESTADO in ESTADOS:
    print("\n" + "=" * 60)
    print(f"  ESTATISTICA DE HOPKINS — {ESTADO} (homicídios confirmados X85–Y09)")
    print("=" * 60)

    df_estado = df_homicidios[df_homicidios["ESTADO"] == ESTADO].copy()
    if df_estado.empty:
        print(f"  Sem dados para {ESTADO}.")
        continue

    X_scaled, features, _ = preparar_features_ohe(df_estado)
    n_disponivel = len(X_scaled)

    # A amostra não pode passar do nº de registros (importa em estados pequenos
    # como RR, AP, AC); usa no máximo 1/3 do estado.
    sample_eff = min(HOPKINS_SAMPLE, max(10, n_disponivel // 3))
    if sample_eff < HOPKINS_SAMPLE:
        print(f"  ⚠ Amostra reduzida para {sample_eff} "
              f"(poucos registros disponíveis: {n_disponivel:,})")

    print(f"  Registros: {n_disponivel:,} | Features (após OHE): {len(features)}")
    print(f"  Calculando Hopkins ({HOPKINS_RUNS} runs x amostra {sample_eff})...")

    resultado = hopkins_statistic(X_scaled, sample_size=sample_eff,
                                  n_runs=HOPKINS_RUNS)
    resultados_hopkins[ESTADO] = resultado

    print(f"\n  H medio : {resultado['media']:.4f}")
    print(f"  H desvio: {resultado['std']:.4f}")
    print(f"  H por run: {[round(v, 4) for v in resultado['valores']]}")
    print(f"\n  {resultado['interpretacao']}")

# ── Figura comparativa: histograma de H por estado, grade 3×3 ─────────────────
_ncols = 3
_nrows = math.ceil(len(ESTADOS) / _ncols)
fig, axes = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 4.2 * _nrows))
axes = np.array(axes).reshape(-1)

fig.suptitle(
    "Estatistica de Hopkins — Tendencia de Clustering\n"
    f"SIM · Homicídios Confirmados X85–Y09 · {' / '.join(ESTADOS)}\n"
    "(espaço de features OHE; H > 0.75 = forte clustering)",
    fontsize=13, fontweight="bold",
)

for idx, ESTADO in enumerate(ESTADOS):
    ax = axes[idx]

    if ESTADO not in resultados_hopkins:
        ax.set_visible(False)
        continue

    res     = resultados_hopkins[ESTADO]
    valores = res["valores"]
    media   = res["media"]
    std     = res["std"]

    ax.hist(valores, bins=8, color=CORES_ESTADO.get(ESTADO, "#6366f1"),
            edgecolor="white", alpha=0.85)
    ax.axvline(media, color="black", linewidth=2,
               linestyle="--", label=f"Media = {media:.4f}")

    # Faixas de referência: aleatório / fraco / forte.
    ax.axvspan(0.0,  0.5,  alpha=0.07, color="red",    label="Aleatorio (< 0.5)")
    ax.axvspan(0.5,  0.75, alpha=0.07, color="orange", label="Fraco (0.5-0.75)")
    ax.axvspan(0.75, 1.0,  alpha=0.07, color="green",  label="Forte (> 0.75)")

    ax.set_title(
        f"{ESTADO}  |  H = {media:.4f} +/- {std:.4f}\n{res['interpretacao']}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Valor de H por run", fontsize=9)
    ax.set_ylabel("Frequencia", fontsize=9)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# Esconde os eixos que sobram na grade (se houver menos estados que células).
for _ax in axes[len(ESTADOS):]:
    _ax.set_visible(False)

plt.tight_layout()
plt.savefig(DIR_FIGURAS + "hopkins_9estados.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSalvo: {DIR_FIGURAS}hopkins_9estados.png")

# ── Resumo final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESUMO — ESTATISTICA DE HOPKINS (homicídios confirmados X85–Y09)")
print("=" * 60)
for ESTADO in ESTADOS:
    if ESTADO in resultados_hopkins:
        res = resultados_hopkins[ESTADO]
        print(f"  {ESTADO}: H = {res['media']:.4f} +/- {res['std']:.4f}"
              f"  |  {res['interpretacao']}")
print("=" * 60)
