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
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage as scipy_linkage

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
AGG_SAMPLE_SIZE = 8_000   # teto: a matriz de distância é O(n²)
K_RANGE         = range(2, 11)
LINKAGE_METHODS = ["ward", "average", "complete"]
DENDRO_LEAVES   = 40
COLUNA_DESTAQUE = "FAIXA_IDADE"


# ── Lógica específica do Aglomerativo (chamada pelo driver, por estado) ───────
def clusteriza_fn(df_estado, ESTADO, ctx):
    X_scaled = ctx["X_scaled"]; features = ctx["features"]
    df_mod   = ctx["df_mod"];   n = ctx["n"]; t_inicio = ctx["t_inicio"]

    print(f"  Espaço de clusterização: OHE direto ({X_scaled.shape[1]}D, sem UMAP)")

    # ── Amostra aleatória (limita o custo O(n²) da matriz de distância) ───────
    sample_size = min(AGG_SAMPLE_SIZE, n)
    rng         = np.random.default_rng(42)
    idx_s       = rng.choice(n, sample_size, replace=False)
    X_agg       = X_scaled[idx_s]
    df_mod_agg  = df_mod.iloc[idx_s].reset_index(drop=True)
    n_agg       = len(X_agg)

    print(f"\n  Amostra aleatória: {n_agg:,} de {n:,}")

    MIN_CLUSTER_FRAC = 0.02   # menor cluster precisa ter ≥ 2% para o K valer

    KS               = list(K_RANGE)
    melhor_global    = {"sil": -np.inf}
    links_por_metodo = {}

    # ── Testa cada método de linkage × cada K, e guarda o melhor por método ───
    for metodo in LINKAGE_METHODS:
        print(f"\n  ── Linkage: {metodo} ──")
        t_l = time.time()
        Z = scipy_linkage(X_agg, method=metodo)
        print(f"     linkage calculado ({time.time()-t_l:.0f}s)")

        resultados   = []
        todos_labels = {}
        for k in KS:
            labels    = fcluster(Z, t=k, criterion="maxclust") - 1
            tamanhos  = np.bincount(labels)
            menor_pct = float(tamanhos.min() / len(labels) * 100)
            metricas  = calcular_metricas_cluster(X_agg, labels)
            resultados.append({
                "K":                 k,
                "Silhouette":        metricas["Silhouette"],
                "Davies-Bouldin":    metricas["Davies-Bouldin"],
                "Dunn_Index":        metricas["Dunn_Index"],
                "menor_cluster_pct": round(menor_pct, 2),
            })
            todos_labels[k] = labels

        df_m = pd.DataFrame(resultados)

        # K elegível = aquele cujo menor cluster passa de MIN_CLUSTER_FRAC; entre os
        # elegíveis, escolhe o de maior Silhouette. Sem nenhum elegível, o método é
        # desqualificado (provável outlier isolado).
        validos_m = df_m[df_m["menor_cluster_pct"] >= MIN_CLUSTER_FRAC * 100]
        validos_m = validos_m.dropna(subset=["Silhouette"])
        elegivel  = not validos_m.empty
        if elegivel:
            idx_best = validos_m["Silhouette"].idxmax()
            k_best   = int(validos_m.loc[idx_best, "K"])
            sil_best = float(validos_m.loc[idx_best, "Silhouette"])
        else:
            k_best   = int(df_m.loc[df_m["Silhouette"].idxmax(), "K"])
            sil_best = float(df_m["Silhouette"].max())
            print(f"     ⚠ Nenhum K teve menor cluster ≥{MIN_CLUSTER_FRAC*100:.0f}% "
                  f"— método DESQUALIFICADO da comparação final "
                  f"(mostrado só para referência; risco de outlier isolado)")

        links_por_metodo[metodo] = {
            "Z": Z, "df_metricas": df_m,
            "todos_labels": todos_labels,
            "k_best": k_best, "sil_best": sil_best,
            "elegivel": elegivel,
        }
        print(f"     melhor K={k_best} | Silhouette={sil_best:.4f}"
              f"{'' if elegivel else '  (desqualificado)'}")

        if elegivel and sil_best > melhor_global["sil"]:
            melhor_global = {
                "sil": sil_best, "metodo": metodo,
                "k": k_best, "Z": Z,
                "labels": todos_labels[k_best],
                "df_metricas": df_m,
            }

    # Se nenhum método ficou elegível, usa o de maior Silhouette só para não vazar.
    if melhor_global["sil"] == -np.inf:
        print(f"\n  ⚠⚠ NENHUM método teve K balanceado — usando o de maior "
              f"Silhouette geral (resultado pode ser pouco interpretável; "
              f"considere reduzir MIN_CLUSTER_FRAC ou revisar os dados).")
        melhor_metodo_fallback = max(
            links_por_metodo, key=lambda m: links_por_metodo[m]["sil_best"]
        )
        info_fb = links_por_metodo[melhor_metodo_fallback]
        melhor_global = {
            "sil": info_fb["sil_best"], "metodo": melhor_metodo_fallback,
            "k": info_fb["k_best"], "Z": info_fb["Z"],
            "labels": info_fb["todos_labels"][info_fb["k_best"]],
            "df_metricas": info_fb["df_metricas"],
        }

    metodo_final = melhor_global["metodo"]
    melhor_k     = melhor_global["k"]
    labels_final = melhor_global["labels"]
    Z_final      = melhor_global["Z"]
    df_metricas  = melhor_global["df_metricas"]

    print(f"\n  → Melhor combinação: linkage={metodo_final}, K={melhor_k} "
          f"(Silhouette={melhor_global['sil']:.4f})")

    # ════════════════════════════════════════════════════════════════════════
    #  FIG 1 — Dendrograma (truncado) do método vencedor
    # ════════════════════════════════════════════════════════════════════════
    # Linha de corte entre as alturas que separam K-1 e K clusters.
    alturas = np.sort(Z_final[:, 2])[::-1]
    if melhor_k > 1 and len(alturas) >= melhor_k:
        color_threshold = (alturas[melhor_k - 2] + alturas[melhor_k - 1]) / 2
    else:
        color_threshold = None

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        Z_final, ax=ax, truncate_mode="lastp", p=DENDRO_LEAVES,
        color_threshold=color_threshold,
        show_leaf_counts=True, leaf_rotation=90, leaf_font_size=7,
    )
    ax.set_title(
        f"Dendrograma (truncado, últimas {DENDRO_LEAVES} fusões) — {ESTADO}\n"
        f"Linkage={metodo_final} | Linha de corte ≈ K={melhor_k} | espaço OHE direto",
        fontweight="bold"
    )
    ax.set_xlabel("Tamanho do cluster (entre parênteses)")
    ax.set_ylabel("Distância")
    plt.tight_layout()
    _png = f"{DIR_FIGURAS}aglomerativo_dendrograma_{ESTADO}.png"
    plt.savefig(_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {_png}")

    # ════════════════════════════════════════════════════════════════════════
    #  FIG 2 — Métricas vs K (método vencedor) + Silhouette por método (painel 4)
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Aglomerativo — {ESTADO} (homicídios X85–Y09 · espaço OHE direto)\n"
        f"Painéis 1-3: método vencedor ({metodo_final}) | Painel 4: Silhouette por método",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    sil_vals = df_metricas["Silhouette"].tolist()
    ax.plot(KS, sil_vals, "o-", color="#10b981", linewidth=2, markersize=7)
    for k, v in zip(KS, sil_vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.axvline(melhor_k, color="orange", linestyle="--", label=f"Melhor K={melhor_k}")
    ax.set_title("Silhouette Score\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    dbi_vals   = df_metricas["Davies-Bouldin"].tolist()
    best_dbi_k = KS[int(np.argmin(dbi_vals))]
    ax.plot(KS, dbi_vals, "o-", color="#ef4444", linewidth=2, markersize=7)
    for k, v in zip(KS, dbi_vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.axvline(best_dbi_k, color="orange", linestyle="--", label=f"Melhor K={best_dbi_k}")
    ax.set_title("Davies-Bouldin Index\n(menor = melhor)", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("DBI")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    dunn_vals   = df_metricas["Dunn_Index"].tolist()
    best_dunn_k = KS[int(np.argmax(dunn_vals))]
    ax.plot(KS, dunn_vals, "o-", color="#f59e0b", linewidth=2, markersize=7)
    for k, v in zip(KS, dunn_vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.axvline(best_dunn_k, color="orange", linestyle="--", label=f"Melhor K={best_dunn_k}")
    ax.set_title("Dunn Index\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Dunn Index")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[3]
    cores_metodo = {"ward": "#6366f1", "average": "#10b981", "complete": "#ef4444"}
    for metodo, info in links_por_metodo.items():
        ax.plot(KS, info["df_metricas"]["Silhouette"], "o-",
                color=cores_metodo.get(metodo, "gray"), linewidth=2,
                markersize=5, label=metodo)
    ax.axvline(melhor_k, color="orange", linestyle="--", alpha=0.7)
    ax.set_title("Silhouette por método de linkage", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    _png = f"{DIR_FIGURAS}aglomerativo_metricas_{ESTADO}.png"
    plt.savefig(_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {_png}")

    # ════════════════════════════════════════════════════════════════════════
    #  Perfil dos clusters
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Perfil dos clusters (linkage={metodo_final}, K={melhor_k}):")
    perfil_df = perfil_clusters(df_mod_agg, labels_final)
    print(perfil_df.to_string())

    # ════════════════════════════════════════════════════════════════════════
    #  UMAP 2D — visualização
    # ════════════════════════════════════════════════════════════════════════
    visualizar_umap(
        X_agg, labels_final,
        titulo=f"Aglomerativo ({metodo_final}) K={melhor_k}",
        estado=ESTADO,
        df_mod=df_mod_agg, coluna_destaque=COLUNA_DESTAQUE,
        n_sample=AGG_SAMPLE_SIZE,
        salvar_como=f"{DIR_FIGURAS}aglomerativo_umap_{ESTADO}.png"
    )

    # ── Tabela resumo ─────────────────────────────────────────────────────────
    t_total = time.time() - t_inicio
    print(f"\n{'═'*70}")
    print(f"  TABELA RESUMO — AGLOMERATIVO — {ESTADO}  (tempo: {t_total/60:.1f} min)")
    print(f"{'═'*70}")
    print(f"  Método de linkage vencedor: {metodo_final}")
    print(df_metricas.to_string(index=False))
    print(f"\n  Silhouette por método: " + " | ".join(
        f"{m}: {info['sil_best']:.4f} (K={info['k_best']})"
        f"{'' if info['elegivel'] else ' [desqualificado]'}"
        for m, info in links_por_metodo.items()
    ))
    print(f"  Melhor K (Silhouette): K = {melhor_k}")
    print(f"  Amostra: {n_agg:,} de {n:,} registros")
    print(f"{'═'*70}\n")

    return df_metricas, {"linkage": metodo_final, "K": melhor_k}, labels_final, perfil_df


# ── Roda o Aglomerativo em todos os estados via driver comum ──────────────────
rodar_pipeline_estado(
    nome_algoritmo="AGLOMERATIVO",
    algo_key="Aglomerativo",
    concluido_msg="Agrupamento Hierárquico Aglomerativo concluído para todos os estados.",
    df_homicidios=df_homicidios,
    estados=ESTADOS,
    clusteriza_fn=clusteriza_fn,
    resultados=resultados_clusterizacao,
    pkl_path=_pkl,
    cache=cache,
)
