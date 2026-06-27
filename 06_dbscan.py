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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from config import COR_BASE, DIR_DADOS, DIR_FIGURAS, DIR_RESULTADOS, ESTADOS
from funcoes import (calcular_metricas_cluster, perfil_clusters,
                     rodar_pipeline_estado, visualizar_umap)
from _cache_features import build_cache

warnings.filterwarnings("ignore")

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
MIN_SAMPLES_DBSCAN    = 10
N_EPS_CANDIDATOS      = 18
MAX_CLUSTERS_DESEJADO = 12
UMAP_SAMPLE           = 10_000
COLUNA_DESTAQUE       = "FAIXA_IDADE"
DBSCAN_MAX_N          = 40_000


# ── Lógica específica do DBSCAN (chamada pelo driver, por estado) ─────────────
def clusteriza_fn(df_estado, ESTADO, ctx):
    X_scaled = ctx["X_scaled"]; features = ctx["features"]
    df_mod   = ctx["df_mod"];   n = ctx["n"]; t_inicio = ctx["t_inicio"]

    print(f"  Espaço de clusterização: OHE direto ({X_scaled.shape[1]}D, sem UMAP)")

    # ── Amostra se o estado for grande demais ─────────────────────────────────
    if n > DBSCAN_MAX_N:
        rng       = np.random.default_rng(42)
        idx_s     = rng.choice(n, DBSCAN_MAX_N, replace=False)
        X_db      = X_scaled[idx_s]
        df_mod_db = df_mod.iloc[idx_s].reset_index(drop=True)
        print(f"  Amostra aleatória: {len(X_db):,} de {n:,} linhas "
              f"(DBSCAN_MAX_N={DBSCAN_MAX_N:,})")
    else:
        X_db      = X_scaled
        df_mod_db = df_mod
        print(f"  Usando todos os {n:,} registros (≤ {DBSCAN_MAX_N:,})")

    n_db = len(X_db)

    # ════════════════════════════════════════════════════════════════════════
    #  Escolha do eps pelo gráfico da k-distância (joelho da curva)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Calculando k-distâncias (k={MIN_SAMPLES_DBSCAN})...", end=" ", flush=True)
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=MIN_SAMPLES_DBSCAN, n_jobs=-1)
    nn.fit(X_db)
    dist_k, _ = nn.kneighbors(X_db)
    k_dist = np.sort(dist_k[:, -1])   # distância ao k-ésimo vizinho, ordenada
    print(f"{time.time()-t0:.0f}s")

    # Joelho = ponto de maior curvatura (2ª diferença máxima) da curva ordenada.
    deltas     = np.diff(k_dist)
    deltas2    = np.diff(deltas)
    idx_joelho = int(np.argmax(deltas2)) + 1 if len(deltas2) > 0 else 0
    eps_sugerido = float(k_dist[idx_joelho])

    # Fallbacks quando há muitos pontos idênticos (k-distância colada em zero).
    if eps_sugerido <= 0:
        eps_sugerido = float(np.percentile(k_dist, 90))
        print(f"  (joelho em zero — muitos pontos duplicados; "
              f"usando percentil 90: {eps_sugerido:.4f})")
    if eps_sugerido <= 0:
        eps_sugerido = float(k_dist.max()) if k_dist.max() > 0 else 1.0
        print(f"  (percentil 90 também em zero — usando máximo: {eps_sugerido:.4f})")

    print(f"  eps sugerido (joelho da curva): {eps_sugerido:.4f}")

    # ── Candidatos de eps: cauda superior da k-distância + alguns percentis ───
    k_dist_unicos = np.unique(k_dist)
    n_tail        = min(len(k_dist_unicos), max(N_EPS_CANDIDATOS * 4, 40))
    tail_vals     = k_dist_unicos[-n_tail:]
    idx_pick      = np.unique(np.linspace(
        0, len(tail_vals) - 1, N_EPS_CANDIDATOS
    ).astype(int))
    eps_criticos  = tail_vals[idx_pick]

    eps_contexto  = [float(np.percentile(k_dist, p)) for p in [50, 70, 85]]

    eps_candidatos = sorted(set(
        round(float(e), 4) for e in (list(eps_criticos) + eps_contexto)
    ))
    eps_candidatos = [e for e in eps_candidatos if e > 0]

    # ════════════════════════════════════════════════════════════════════════
    #  Testa cada eps e guarda as métricas
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Testando {len(eps_candidatos)} valores de eps "
          f"(min_samples={MIN_SAMPLES_DBSCAN})...")

    resultados   = []
    todos_labels = {}

    for eps in eps_candidatos:
        t_e = time.time()
        db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES_DBSCAN, n_jobs=-1)
        labels = db.fit_predict(X_db)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int((labels == -1).sum())
        pct_noise  = n_noise / n_db * 100

        metricas = calcular_metricas_cluster(X_db, labels)

        resultados.append({
            "eps":            eps,
            "n_clusters":     n_clusters,
            "n_noise":        n_noise,
            "%_noise":        round(pct_noise, 2),
            "Silhouette":     metricas["Silhouette"],
            "Davies-Bouldin": metricas["Davies-Bouldin"],
            "Dunn_Index":     metricas["Dunn_Index"],
        })
        todos_labels[eps] = labels

        sil_str = (f"{metricas['Silhouette']:.4f}"
                   if not np.isnan(metricas["Silhouette"]) else "  n/a")
        print(f"    eps={eps:8.4f} | clusters={n_clusters:3d} | "
              f"ruído={pct_noise:5.1f}% | Sil={sil_str} | {time.time()-t_e:.0f}s")

    df_metricas = pd.DataFrame(resultados)

    # ── Escolha do melhor eps ─────────────────────────────────────────────────
    # Preferência: maior Silhouette entre as soluções "razoáveis" (2..12 clusters,
    # ruído < 50%).
    validos = df_metricas[
        (df_metricas["n_clusters"] >= 2) &
        (df_metricas["n_clusters"] <= MAX_CLUSTERS_DESEJADO) &
        (df_metricas["%_noise"] < 50)
    ].dropna(subset=["Silhouette"])

    if not validos.empty:
        melhor_eps = validos.loc[validos["Silhouette"].idxmax(), "eps"]
    else:
        # Nenhuma solução razoável: pega a menos fragmentada que ainda tenha ruído
        # < 50%, e avisa que pode valer testar eps maiores.
        usaveis = df_metricas[
            (df_metricas["n_clusters"] >= 2) & (df_metricas["%_noise"] < 50)
        ].dropna(subset=["Silhouette"])
        if usaveis.empty:
            print("\n  ⚠ Nenhuma combinação produziu ≥2 clusters com ruído < 50%.")
            print("    Considere ajustar MIN_SAMPLES_DBSCAN ou N_EPS_CANDIDATOS.")
            melhor_eps = eps_candidatos[len(eps_candidatos) // 2]
        else:
            melhor_eps = usaveis.loc[usaveis["n_clusters"].idxmin(), "eps"]
            n_esc = int(usaveis.loc[usaveis["n_clusters"].idxmin(), "n_clusters"])
            print(f"\n  ⚠ Nenhum eps ficou ≤ {MAX_CLUSTERS_DESEJADO} clusters; "
                  f"escolhido o de menor fragmentação disponível "
                  f"(eps={melhor_eps}, {n_esc} clusters). Considere testar "
                  f"valores de eps ainda maiores (ajuste N_EPS_CANDIDATOS "
                  f"ou o intervalo de percentis).")

    labels_final = todos_labels[melhor_eps]
    print(f"\n  → eps escolhido: {melhor_eps}")

    # ════════════════════════════════════════════════════════════════════════
    #  FIG 1 — k-distância + métricas (Silhouette/DBI/Dunn) e nº clusters vs eps
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 5, figsize=(27, 5))
    fig.suptitle(
        f"DBSCAN — Seleção de eps | SIM · {ESTADO} (homicídios confirmados X85–Y09)",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(k_dist, color=COR_BASE, linewidth=1.5)
    ax.axhline(eps_sugerido, color="orange", linestyle="--",
               label=f"Sugerido: {eps_sugerido:.3f}")
    ax.set_title(f"Gráfico k-distância (k={MIN_SAMPLES_DBSCAN})", fontweight="bold")
    ax.set_xlabel("Pontos (ordenados)")
    ax.set_ylabel(f"Dist. ao {MIN_SAMPLES_DBSCAN}º vizinho")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    sil_vals = df_metricas["Silhouette"].tolist()
    ax.plot(df_metricas["eps"], sil_vals, "o-", color="#10b981",
            linewidth=2, markersize=7)
    ax.axvline(melhor_eps, color="orange", linestyle="--",
               label=f"Escolhido eps={melhor_eps}")
    ax.set_title("Silhouette vs eps\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("eps"); ax.set_ylabel("Silhouette")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(df_metricas["eps"], df_metricas["Davies-Bouldin"], "o-",
            color="#ef4444", linewidth=2, markersize=7)
    ax.axvline(melhor_eps, color="orange", linestyle="--")
    ax.set_title("Davies-Bouldin vs eps\n(menor = melhor)", fontweight="bold")
    ax.set_xlabel("eps"); ax.set_ylabel("DBI")
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(df_metricas["eps"], df_metricas["Dunn_Index"], "o-",
            color="#f59e0b", linewidth=2, markersize=7)
    ax.axvline(melhor_eps, color="orange", linestyle="--")
    ax.set_title("Dunn Index vs eps\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("eps"); ax.set_ylabel("Dunn Index")
    ax.grid(alpha=0.3)

    # Eixo duplo: nº de clusters (esquerda) e % de ruído (direita) lado a lado.
    ax  = axes[4]
    ax2 = ax.twinx()
    ax.plot(df_metricas["eps"], df_metricas["n_clusters"], "o-",
            color="#6366f1", linewidth=2, label="n_clusters")
    ax2.plot(df_metricas["eps"], df_metricas["%_noise"], "s--",
             color="gray", linewidth=2, label="% ruído")
    ax.axvline(melhor_eps, color="orange", linestyle="--")
    ax.set_title("Nº Clusters e % Ruído vs eps", fontweight="bold")
    ax.set_xlabel("eps")
    ax.set_ylabel("Nº clusters", color="#6366f1")
    ax2.set_ylabel("% ruído", color="gray")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _png = f"{DIR_FIGURAS}dbscan_metricas_{ESTADO}.png"
    plt.savefig(_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {_png}")

    # ════════════════════════════════════════════════════════════════════════
    #  Perfil dos clusters
    # ════════════════════════════════════════════════════════════════════════
    n_clusters_final = len(set(labels_final)) - (1 if -1 in labels_final else 0)
    print(f"\n  Perfil dos clusters "
          f"(eps={melhor_eps}, {n_clusters_final} clusters + ruído):")
    perfil_df = perfil_clusters(df_mod_db, labels_final)
    print(perfil_df.to_string())

    # ════════════════════════════════════════════════════════════════════════
    #  UMAP 2D — visualização
    # ════════════════════════════════════════════════════════════════════════
    visualizar_umap(
        X_db, labels_final,
        titulo=f"DBSCAN eps={melhor_eps}",
        estado=ESTADO,
        df_mod=df_mod_db, coluna_destaque=COLUNA_DESTAQUE,
        n_sample=UMAP_SAMPLE,
        salvar_como=f"{DIR_FIGURAS}dbscan_umap_{ESTADO}.png"
    )

    # ── Tabela resumo ─────────────────────────────────────────────────────────
    t_total = time.time() - t_inicio
    print(f"\n{'═'*70}")
    print(f"  TABELA RESUMO — DBSCAN — {ESTADO}  (tempo: {t_total/60:.1f} min)")
    print(f"{'═'*70}")
    print(df_metricas.to_string(index=False))
    print(f"\n  eps escolhido: {melhor_eps}  |  min_samples: {MIN_SAMPLES_DBSCAN}")
    print(f"{'═'*70}\n")

    return df_metricas, melhor_eps, labels_final, perfil_df


# ── Roda o DBSCAN em todos os estados via driver comum ────────────────────────
rodar_pipeline_estado(
    nome_algoritmo="DBSCAN",
    algo_key="DBSCAN",
    concluido_msg="DBSCAN concluído para todos os estados.",
    df_homicidios=df_homicidios,
    estados=ESTADOS,
    clusteriza_fn=clusteriza_fn,
    resultados=resultados_clusterizacao,
    pkl_path=_pkl,
    cache=cache,
)
