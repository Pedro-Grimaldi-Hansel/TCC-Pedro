import matplotlib
matplotlib.use('Agg')

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan as _hdb

from config import COR_BASE, DIR_DADOS, DIR_FIGURAS, DIR_RESULTADOS, ESTADOS
from funcoes import (calcular_metricas_cluster, perfil_clusters,
                     rodar_pipeline_estado, visualizar_umap)
from _cache_features import build_cache

warnings.filterwarnings("ignore")

# Pacote standalone (não o sklearn.cluster.HDBSCAN), que conflita com numpy 2.x.
HDBSCAN = _hdb.HDBSCAN

# ── Carrega dados ─────────────────────────────────────────────────────────────
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)
df_homicidios = pd.read_parquet(DIR_DADOS + "homicidios.parquet")

# ── Cache de features OHE (compartilhado entre 05–08 via disco) ───────────────
cache = build_cache(df_homicidios)

# ── Resultados acumuladores (05–08 vão somando suas chaves no mesmo pickle) ───
_pkl = DIR_RESULTADOS + "resultados.pkl"
try:
    with open(_pkl, "rb") as _f:
        resultados_clusterizacao = pickle.load(_f)
except FileNotFoundError:
    resultados_clusterizacao = {estado: {} for estado in ESTADOS}

# ── Parâmetros ────────────────────────────────────────────────────────────────
MIN_CLUSTER_SIZE_PCT = 0.01
# Varredura de cluster_selection_epsilon: epsilons maiores fundem clusters próximos
# (menos clusters, menos ruído).
CLUSTER_SELECTION_EPSILONS = [
    0.0, 0.2, 0.4, 0.6, 0.8,
    0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15,
    1.2, 1.5, 2.0, 2.5,
]
MAX_CLUSTERS_DESEJADO = 12
UMAP_SAMPLE           = 10_000
COLUNA_DESTAQUE       = "FAIXA_IDADE"
HDBSCAN_MAX_N         = 40_000


# ── Lógica específica do HDBSCAN (chamada pelo driver, por estado) ────────────
def clusteriza_fn(df_estado, ESTADO, ctx):
    X_scaled = ctx["X_scaled"]; features = ctx["features"]
    df_mod   = ctx["df_mod"];   n = ctx["n"]; t_inicio = ctx["t_inicio"]

    print(f"  Espaço de clusterização: OHE direto ({X_scaled.shape[1]}D, sem UMAP)")

    # ── Amostra se o estado for grande demais ─────────────────────────────────
    if n > HDBSCAN_MAX_N:
        rng       = np.random.default_rng(42)
        idx_s     = rng.choice(n, HDBSCAN_MAX_N, replace=False)
        X_hd      = X_scaled[idx_s]
        df_mod_hd = df_mod.iloc[idx_s].reset_index(drop=True)
        print(f"  Amostra aleatória: {len(X_hd):,} de {n:,} linhas "
              f"(HDBSCAN_MAX_N={HDBSCAN_MAX_N:,})")
    else:
        X_hd      = X_scaled
        df_mod_hd = df_mod
        print(f"  Usando todos os {n:,} registros (≤ {HDBSCAN_MAX_N:,})")

    n_hd = len(X_hd)
    # min_cluster_size fixo: ~1% do estado, com um piso de 20 para estados pequenos.
    min_cluster_size_fixo = max(20, int(MIN_CLUSTER_SIZE_PCT * n_hd))

    print(f"  min_cluster_size fixo: {min_cluster_size_fixo} "
          f"({MIN_CLUSTER_SIZE_PCT*100:.0f}% de {n_hd:,})")

    # ════════════════════════════════════════════════════════════════════════
    #  Varre cluster_selection_epsilon (min_cluster_size fixo)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Testando cluster_selection_epsilon = "
          f"{CLUSTER_SELECTION_EPSILONS}...")

    resultados   = []
    todos_labels = {}

    for csl_eps in CLUSTER_SELECTION_EPSILONS:
        t_c = time.time()
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size_fixo,
            cluster_selection_epsilon=csl_eps,
        )
        labels = clusterer.fit_predict(X_hd)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int((labels == -1).sum())
        pct_noise  = n_noise / n_hd * 100

        metricas = calcular_metricas_cluster(X_hd, labels)

        resultados.append({
            "cluster_selection_epsilon": csl_eps,
            "n_clusters":                n_clusters,
            "n_noise":                   n_noise,
            "%_noise":                   round(pct_noise, 2),
            "Silhouette":                metricas["Silhouette"],
            "Davies-Bouldin":            metricas["Davies-Bouldin"],
            "Dunn_Index":                metricas["Dunn_Index"],
        })
        todos_labels[csl_eps] = labels

        sil_str = (f"{metricas['Silhouette']:.4f}"
                   if not np.isnan(metricas["Silhouette"]) else "  n/a")
        print(f"    cluster_selection_epsilon={csl_eps:4.2f} | "
              f"clusters={n_clusters:3d} | ruído={pct_noise:5.1f}% | "
              f"Sil={sil_str} | {time.time()-t_c:.0f}s")

    df_metricas = pd.DataFrame(resultados)

    # ── Escolha do melhor epsilon (mesmo critério do DBSCAN) ──────────────────
    # Maior Silhouette entre as soluções razoáveis (2..12 clusters, ruído < 50%).
    validos = df_metricas[
        (df_metricas["n_clusters"] >= 2) &
        (df_metricas["n_clusters"] <= MAX_CLUSTERS_DESEJADO) &
        (df_metricas["%_noise"] < 50)
    ].dropna(subset=["Silhouette"])

    if not validos.empty:
        melhor_csl = float(
            validos.loc[validos["Silhouette"].idxmax(), "cluster_selection_epsilon"]
        )
    else:
        # Sem solução razoável: pega a menos fragmentada com ruído < 50%.
        usaveis = df_metricas[
            (df_metricas["n_clusters"] >= 2) & (df_metricas["%_noise"] < 50)
        ].dropna(subset=["Silhouette"])
        if usaveis.empty:
            print("\n  ⚠ Nenhum cluster_selection_epsilon produziu ≥2 "
                  "clusters com ruído < 50%.")
            melhor_csl = CLUSTER_SELECTION_EPSILONS[0]
        else:
            melhor_csl = float(
                usaveis.loc[usaveis["n_clusters"].idxmin(), "cluster_selection_epsilon"]
            )
            n_esc = int(usaveis.loc[usaveis["n_clusters"].idxmin(), "n_clusters"])
            print(f"\n  ⚠ Nenhum cluster_selection_epsilon ficou ≤ "
                  f"{MAX_CLUSTERS_DESEJADO} clusters; escolhido o de menor "
                  f"fragmentação disponível (epsilon={melhor_csl}, "
                  f"{n_esc} clusters). Considere testar valores maiores em "
                  f"CLUSTER_SELECTION_EPSILONS.")

    labels_final = todos_labels[melhor_csl]
    print(f"\n  → cluster_selection_epsilon escolhido: {melhor_csl}")

    # ════════════════════════════════════════════════════════════════════════
    #  FIG 1 — Métricas e nº clusters/ruído vs cluster_selection_epsilon
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"HDBSCAN — Métricas vs cluster_selection_epsilon "
        f"(min_cluster_size={min_cluster_size_fixo}) | "
        f"SIM · {ESTADO} (homicídios confirmados X85–Y09)",
        fontsize=13, fontweight="bold"
    )

    csl_vals = df_metricas["cluster_selection_epsilon"].tolist()

    ax = axes[0]
    ax.plot(csl_vals, df_metricas["Silhouette"], "o-",
            color="#10b981", linewidth=2, markersize=7)
    for c, v in zip(csl_vals, df_metricas["Silhouette"]):
        if not np.isnan(v):
            ax.annotate(f"{v:.3f}", (c, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5)
    ax.axvline(melhor_csl, color="orange", linestyle="--",
               label=f"Escolhido={melhor_csl}")
    ax.set_title("Silhouette\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("cluster_selection_epsilon"); ax.set_ylabel("Silhouette")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(csl_vals, df_metricas["Davies-Bouldin"], "o-",
            color="#ef4444", linewidth=2, markersize=7)
    ax.axvline(melhor_csl, color="orange", linestyle="--")
    ax.set_title("Davies-Bouldin\n(menor = melhor)", fontweight="bold")
    ax.set_xlabel("cluster_selection_epsilon"); ax.set_ylabel("DBI")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(csl_vals, df_metricas["Dunn_Index"], "o-",
            color="#f59e0b", linewidth=2, markersize=7)
    ax.axvline(melhor_csl, color="orange", linestyle="--")
    ax.set_title("Dunn Index\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("cluster_selection_epsilon"); ax.set_ylabel("Dunn Index")
    ax.grid(alpha=0.3)

    # Eixo duplo: nº de clusters (esquerda) e % de ruído (direita).
    ax  = axes[3]
    ax2 = ax.twinx()
    ax.plot(csl_vals, df_metricas["n_clusters"], "o-",
            color="#6366f1", linewidth=2, label="n_clusters")
    ax2.plot(csl_vals, df_metricas["%_noise"], "s--",
             color="gray", linewidth=2, label="% ruído")
    ax.axvline(melhor_csl, color="orange", linestyle="--")
    ax.set_title("Nº Clusters e % Ruído", fontweight="bold")
    ax.set_xlabel("cluster_selection_epsilon")
    ax.set_ylabel("Nº clusters", color="#6366f1")
    ax2.set_ylabel("% ruído", color="gray")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _png = f"{DIR_FIGURAS}hdbscan_metricas_{ESTADO}.png"
    plt.savefig(_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {_png}")

    # ════════════════════════════════════════════════════════════════════════
    #  Perfil dos clusters
    # ════════════════════════════════════════════════════════════════════════
    n_clusters_final = len(set(labels_final)) - (1 if -1 in labels_final else 0)
    print(f"\n  Perfil dos clusters "
          f"(cluster_selection_epsilon={melhor_csl}, "
          f"{n_clusters_final} clusters + ruído):")
    perfil_df = perfil_clusters(df_mod_hd, labels_final)
    print(perfil_df.to_string())

    # ════════════════════════════════════════════════════════════════════════
    #  UMAP 2D — visualização
    # ════════════════════════════════════════════════════════════════════════
    visualizar_umap(
        X_hd, labels_final,
        titulo=f"HDBSCAN cluster_selection_epsilon={melhor_csl}",
        estado=ESTADO,
        df_mod=df_mod_hd, coluna_destaque=COLUNA_DESTAQUE,
        n_sample=UMAP_SAMPLE,
        salvar_como=f"{DIR_FIGURAS}hdbscan_umap_{ESTADO}.png"
    )

    # ── Tabela resumo ─────────────────────────────────────────────────────────
    t_total = time.time() - t_inicio
    print(f"\n{'═'*70}")
    print(f"  TABELA RESUMO — HDBSCAN — {ESTADO}  (tempo: {t_total/60:.1f} min)")
    print(f"{'═'*70}")
    print(df_metricas.to_string(index=False))
    print(f"\n  min_cluster_size fixo: {min_cluster_size_fixo}")
    print(f"  cluster_selection_epsilon escolhido: {melhor_csl}")
    print(f"{'═'*70}\n")

    melhor_param = {
        "min_cluster_size": min_cluster_size_fixo,
        "cluster_selection_epsilon": melhor_csl,
    }
    return df_metricas, melhor_param, labels_final, perfil_df


# ── Roda o HDBSCAN em todos os estados via driver comum ───────────────────────
rodar_pipeline_estado(
    nome_algoritmo="HDBSCAN",
    algo_key="HDBSCAN",
    concluido_msg="HDBSCAN concluído para todos os estados.",
    df_homicidios=df_homicidios,
    estados=ESTADOS,
    clusteriza_fn=clusteriza_fn,
    resultados=resultados_clusterizacao,
    pkl_path=_pkl,
    cache=cache,
)
