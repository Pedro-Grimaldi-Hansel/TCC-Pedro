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
from sklearn.cluster import KMeans

from config import COR_BASE, DIR_DADOS, DIR_FIGURAS, DIR_RESULTADOS, ESTADOS
from funcoes import (calcular_metricas_cluster, perfil_clusters,
                     visualizar_umap)
from _cache_features import build_cache

warnings.filterwarnings("ignore")

# ── Carrega dados ─────────────────────────────────────────────────────────────
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)
df_homicidios = pd.read_parquet(DIR_DADOS + "homicidios.parquet")

# ── Cache de features OHE: construído aqui e reaproveitado por 06–08 via disco ─
cache = build_cache(df_homicidios)

# ── Resultados acumuladores: carrega o que já existe ou começa um dict vazio ──
_pkl = DIR_RESULTADOS + "resultados.pkl"
try:
    with open(_pkl, "rb") as _f:
        resultados_clusterizacao = pickle.load(_f)
except FileNotFoundError:
    resultados_clusterizacao = {estado: {} for estado in ESTADOS}

# ── Parâmetros ────────────────────────────────────────────────────────────────
SIL_SAMPLE_SIZE = 15_000
UMAP_SAMPLE     = 10_000
COLUNA_DESTAQUE = "FAIXA_IDADE"
K_RANGE         = range(2, 11)
K_FINAL         = 7        # premissa de comparabilidade (convergência Dunn em MT/AM)
N_INIT          = 10
MAX_ITER        = 300

# Guarda o K do pico de Dunn de cada estado, para verificar a premissa K=7 no fim.
dunn_peak_por_estado = {}

# ── Loop por estado ───────────────────────────────────────────────────────────
for ESTADO in ESTADOS:
    t_inicio = time.time()
    print("\n" + "█"*70)
    print(f"  K-MEANS — ESTADO: {ESTADO}")
    print("█"*70)

    df_estado = df_homicidios[df_homicidios["ESTADO"] == ESTADO].copy()
    if df_estado.empty:
        print(f"  Sem dados para {ESTADO}. Pulando.")
        continue

    print(f"  Registros (homicídios confirmados X85–Y09): {len(df_estado):,}")
    X_scaled, features, df_mod = cache[ESTADO]
    n = len(X_scaled)
    print(f"  Features após OHE: {len(features)} | "
          f"Espaço de clusterização: OHE direto ({X_scaled.shape[1]}D, sem UMAP)")

    # ── Roda K-Means para cada K e coleta as métricas ────────────────────────
    print(f"\n  K=2..10 (n_init={N_INIT})...")
    wcss_lst, resultados, todos_modelos = [], [], {}
    KS = list(K_RANGE)

    for k in KS:
        t_k = time.time()
        km = KMeans(n_clusters=k, n_init=N_INIT, max_iter=MAX_ITER, random_state=42)
        labels = km.fit_predict(X_scaled)
        wcss_lst.append(km.inertia_)

        metricas = calcular_metricas_cluster(X_scaled, labels, SIL_SAMPLE_SIZE)
        resultados.append({
            "K": k, "WCSS": round(km.inertia_, 1),
            "Silhouette": metricas["Silhouette"],
            "Davies-Bouldin": metricas["Davies-Bouldin"],
            "Dunn_Index": metricas["Dunn_Index"],
        })
        todos_modelos[k] = {"labels": labels}

        marcador = "  ← K final" if k == K_FINAL else ""
        print(f"    K={k:2d} | Sil={metricas['Silhouette']:.4f} | "
              f"DBI={metricas['Davies-Bouldin']:.4f} | "
              f"Dunn={metricas['Dunn_Index']:.4f} | {time.time()-t_k:.0f}s{marcador}")

    df_metricas = pd.DataFrame(resultados)

    # ── K sugerido por cada critério ──────────────────────────────────────────
    # Cotovelo = ponto de maior "dobra" da curva WCSS (2ª diferença máxima).
    deltas     = np.diff(wcss_lst)
    deltas2    = np.diff(deltas)
    cotovelo_k = KS[1:-1][int(np.argmax(np.abs(deltas2)))] if len(deltas2) > 0 else KS[0]

    melhor_k_sil  = int(df_metricas.loc[df_metricas["Silhouette"].idxmax(), "K"])
    melhor_k_dbi  = int(df_metricas.loc[df_metricas["Davies-Bouldin"].idxmin(), "K"])
    melhor_k_dunn = int(df_metricas.loc[df_metricas["Dunn_Index"].idxmax(), "K"])

    dunn_peak_por_estado[ESTADO] = melhor_k_dunn

    # K adotado para os rótulos finais deste estado: o pico de Dunn.
    melhor_k     = dunn_peak_por_estado.get(ESTADO, K_FINAL)
    labels_final = todos_modelos[melhor_k]["labels"]

    print(f"\n  → Cotovelo sugerido:           K = {cotovelo_k}")
    print(f"  → Melhor K por Silhouette:     K = {melhor_k_sil}")
    print(f"  → Melhor K por Davies-Bouldin: K = {melhor_k_dbi}")
    print(f"  → Melhor K por Dunn Index:     K = {melhor_k_dunn}")
    print(f"  → K FINAL ADOTADO:             K = {melhor_k}")
    if melhor_k_dunn != K_FINAL:
        print(f"  ⚠ Atenção: pico do Dunn deste estado é K={melhor_k_dunn}, "
              f"não K={K_FINAL}. (verificação consolidada no fim da célula)")

    # ════════════════════════════════════════════════════════════════════════
    #  FIG 1 — WCSS (cotovelo) + Silhouette + Davies-Bouldin + Dunn vs K
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"K-Means — Métricas vs K | SIM · {ESTADO} (homicídios confirmados X85–Y09)\n"
        f"Linha cinza sólida = K final adotado (K={K_FINAL})",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(KS, wcss_lst, "o-", color=COR_BASE, linewidth=2, markersize=7)
    ax.axvline(cotovelo_k, color="orange", linestyle="--", linewidth=1.8,
               label=f"Cotovelo K={cotovelo_k}")
    ax.axvline(K_FINAL, color="gray", linestyle="-", linewidth=2, alpha=0.7,
               label=f"K final={K_FINAL}")
    ax.set_title("Método do Cotovelo\n(WCSS)", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("WCSS (Inércia)")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    sil_vals = df_metricas["Silhouette"].tolist()
    ax.plot(KS, sil_vals, "o-", color="#10b981", linewidth=2, markersize=7)
    for k, v in zip(KS, sil_vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.axvline(melhor_k_sil, color="orange", linestyle="--", linewidth=1.8,
               label=f"Melhor K={melhor_k_sil}")
    ax.axvline(K_FINAL, color="gray", linestyle="-", linewidth=2, alpha=0.7,
               label=f"K final={K_FINAL}")
    ax.set_title("Silhouette Score\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    dbi_vals = df_metricas["Davies-Bouldin"].tolist()
    ax.plot(KS, dbi_vals, "o-", color="#ef4444", linewidth=2, markersize=7)
    for k, v in zip(KS, dbi_vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.axvline(melhor_k_dbi, color="orange", linestyle="--", linewidth=1.8,
               label=f"Melhor K={melhor_k_dbi}")
    ax.axvline(K_FINAL, color="gray", linestyle="-", linewidth=2, alpha=0.7,
               label=f"K final={K_FINAL}")
    ax.set_title("Davies-Bouldin Index\n(menor = melhor)", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("DBI")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[3]
    dunn_vals = df_metricas["Dunn_Index"].tolist()
    ax.plot(KS, dunn_vals, "o-", color="#f59e0b", linewidth=2, markersize=7)
    for k, v in zip(KS, dunn_vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.axvline(melhor_k_dunn, color="orange", linestyle="--", linewidth=1.8,
               label=f"Melhor K={melhor_k_dunn}")
    ax.axvline(K_FINAL, color="gray", linestyle="-", linewidth=2, alpha=0.7,
               label=f"K final={K_FINAL}")
    ax.set_title("Dunn Index\n(maior = melhor)", fontweight="bold")
    ax.set_xlabel("K"); ax.set_ylabel("Dunn Index")
    ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    _png = f"{DIR_FIGURAS}kmeans_metricas_{ESTADO}.png"
    plt.savefig(_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {_png}")

    # ════════════════════════════════════════════════════════════════════════
    #  FIG 2 — Heatmap das métricas normalizadas (verde = melhor para todas)
    # ════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        f"Heatmap — Métricas Normalizadas | {ESTADO} — K-Means\n"
        "(verde = melhor | DBI já invertido)",
        fontsize=12, fontweight="bold"
    )
    mn_ = df_metricas.set_index("K")[
        ["Silhouette", "Davies-Bouldin", "Dunn_Index"]
    ].copy()
    # Normaliza cada métrica para [0, 1]; o DBI é invertido (menor é melhor) para
    # que "verde = melhor" valha para as três.
    for col in mn_.columns:
        mn, mx = mn_[col].min(), mn_[col].max()
        if mx == mn:
            mn_[col] = 0.5
        elif col == "Davies-Bouldin":
            mn_[col] = 1 - (mn_[col] - mn) / (mx - mn)
        else:
            mn_[col] = (mn_[col] - mn) / (mx - mn)
    sns.heatmap(mn_, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "Score normalizado (0–1, maior = melhor)"})
    ax.set_xlabel("Métrica"); ax.set_ylabel("K")
    plt.tight_layout()
    _png = f"{DIR_FIGURAS}kmeans_heatmap_{ESTADO}.png"
    plt.savefig(_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {_png}")

    # ── Perfil dos clusters + UMAP no K adotado ──────────────────────────────
    print(f"\n  Perfil dos clusters (K={melhor_k}):")
    perfil_df = perfil_clusters(df_mod, labels_final)
    print(perfil_df.to_string())

    visualizar_umap(
        X_scaled, labels_final,
        titulo=f"K-Means K={melhor_k}", estado=ESTADO,
        df_mod=df_mod, coluna_destaque=COLUNA_DESTAQUE,
        n_sample=UMAP_SAMPLE,
        salvar_como=f"{DIR_FIGURAS}kmeans_umap_{ESTADO}.png"
    )

    t_total = time.time() - t_inicio
    print(f"\n{'═'*70}")
    print(f"  TABELA RESUMO — {ESTADO}  (tempo: {t_total/60:.1f} min)")
    print(f"{'═'*70}")
    print(df_metricas.to_string(index=False))
    print(f"\n  Cotovelo: K={cotovelo_k} | Sil: K={melhor_k_sil} | "
          f"DBI: K={melhor_k_dbi} | Dunn: K={melhor_k_dunn}")
    print(f"  K adotado neste estado (pico Dunn): K = {melhor_k}")
    print(f"{'═'*70}\n")

    resultados_clusterizacao[ESTADO]["KMeans"] = {
        "df_metricas": df_metricas,
        "melhor_param": melhor_k,
        "labels": labels_final,
        "perfil": perfil_df,
    }

    # Bootstrap (caro) só nos estados menores, como amostra de estabilidade.
    if len(X_scaled) < 6000:
        print(f"  Calculando estabilidade bootstrap (n_bootstrap=20)...")
        from funcoes import estabilidade_bootstrap
        boot = estabilidade_bootstrap(X_scaled, k=melhor_k, n_bootstrap=20)
        print(f"  ARI bootstrap: {boot['media']:.4f} ± {boot['std']:.4f}")
        print(f"  {'ESTÁVEL' if boot['media'] > 0.6 else 'INSTÁVEL — interpretar com cautela'}")
        resultados_clusterizacao[ESTADO]["KMeans"]["bootstrap"] = boot

K_FINAL_POR_ESTADO = {estado: dunn_peak_por_estado.get(estado, K_FINAL)
                      for estado in ESTADOS}

# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICAÇÃO DA PREMISSA K_FINAL=7 — o pico de Dunn realmente fica em 7?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print(f"  VERIFICAÇÃO DA PREMISSA K_FINAL = {K_FINAL} (pico do Dunn por estado)")
print("═"*70)
if dunn_peak_por_estado:
    concordam = [e for e, k in dunn_peak_por_estado.items() if k == K_FINAL]
    divergem  = {e: k for e, k in dunn_peak_por_estado.items() if k != K_FINAL}
    for e in ESTADOS:
        if e in dunn_peak_por_estado:
            k = dunn_peak_por_estado[e]
            flag = "✓ K=7" if k == K_FINAL else f"✗ pico em K={k}"
            print(f"    {e}: Dunn máximo em K={k}   {flag}")
    n_tot = len(dunn_peak_por_estado)
    print(f"\n  Concordam com K={K_FINAL}: {len(concordam)}/{n_tot} estados "
          f"({', '.join(concordam) if concordam else '—'})")
    if divergem:
        print(f"  Divergem: {divergem}")
    # Se menos da metade concorda, a premissa global K=7 não se sustenta.
    if len(concordam) < n_tot / 2:
        print(f"\n  ⚠⚠ A MAIORIA dos estados NÃO tem pico de Dunn em K={K_FINAL}. "
              f"A premissa herdada de MT/AM NÃO se generaliza para os 9 estados — "
              f"recomendo reavaliar K_FINAL (ex.: usar o K modal do pico de Dunn, "
              f"ou um K por estado em vez de um K global).")
    else:
        print(f"\n  ✓ K={K_FINAL} continua sendo escolha razoável "
              f"(maioria dos estados concorda ou fica adjacente).")
print("═"*70)

# ── Persiste resultados ───────────────────────────────────────────────────────
with open(_pkl, "wb") as _f:
    pickle.dump(resultados_clusterizacao, _f)
print(f"\n  Salvo: {_pkl}")

print("\nK-Means concluído para todos os estados.")
