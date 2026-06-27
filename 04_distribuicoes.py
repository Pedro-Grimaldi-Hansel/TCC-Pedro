import matplotlib
matplotlib.use('Agg')

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from config import CORES_ESTADO, DIR_DADOS, DIR_FIGURAS, DIR_RESULTADOS, ESTADOS
from funcoes import preparar_features

# ── Carrega dados ─────────────────────────────────────────────────────────────
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)
df_homicidios = pd.read_parquet(DIR_DADOS + "homicidios.parquet")

# ── Rótulos legíveis das features e suas categorias ───────────────────────────

LABELS_FEATURES = {
    "IDADE_NUM":   "Idade (anos)",
    "HORA_NUM":    "Hora do Óbito",
    "ANO_NUM":     "Ano",
    "SEXO_NUM":    "Sexo",
    "RACACOR_NUM": "Raça/Cor",
    "ESTCIV_NUM":  "Estado Civil",
    "ESC_NUM":     "Escolaridade",
    "LOCOCOR_NUM": "Local da Ocorrência",
}

# Contínuas usam KDE; categóricas usam perfil % por categoria (ver funções abaixo).
FEATURES_CONTINUAS   = ["IDADE_NUM", "HORA_NUM", "ANO_NUM"]
FEATURES_CATEGORICAS = ["SEXO_NUM", "RACACOR_NUM", "ESTCIV_NUM",
                        "ESC_NUM", "LOCOCOR_NUM"]

# Tradução código → rótulo para o eixo x dos gráficos categóricos.
ROTULOS_CATEGORICOS = {
    "SEXO_NUM": {
        1: "Masculino", 2: "Feminino", 0: "Ignorado",
    },
    "RACACOR_NUM": {
        1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda",
        5: "Indígena", 0: "Ignorado",
    },
    "ESTCIV_NUM": {
        1: "Solteiro", 2: "Casado", 3: "Viúvo", 4: "Separado",
        5: "União Estável", 0: "Ignorado",
    },
    "ESC_NUM": {
        1: "Nenhuma", 2: "1 a 3 anos", 3: "4 a 7 anos",
        4: "8 a 11 anos", 5: "12 anos e mais", 0: "Ignorado",
    },
    "LOCOCOR_NUM": {
        1: "Hospital", 2: "Outros Estab. Saúde", 3: "Domicílio",
        4: "Via Pública", 5: "Outros", 0: "Ignorado",
    },
}

# ── Prepara os dados de cada estado (via descritiva, numérica) ────────────────
dados_por_estado = {}
features_comuns  = None

for ESTADO in ESTADOS:
    df_estado = df_homicidios[df_homicidios["ESTADO"] == ESTADO].copy()
    if df_estado.empty:
        print(f"  Sem dados para {ESTADO}.")
        continue
    _, _, features, df_modelo = preparar_features(df_estado)
    dados_por_estado[ESTADO] = df_modelo
    features_comuns = features
    print(f"  {ESTADO}: {len(df_modelo):,} homicídios confirmados (X85–Y09)")

estados_com_dados = [e for e in ESTADOS if e in dados_por_estado]


# ── Funções locais ────────────────────────────────────────────────────────────

def linha_continua_feature(feat, dados_por_estado, estados, cores, label,
                           n_pts=400, salvar_como=None):
    """Sobrepõe uma curva de densidade (KDE) por estado, no mesmo eixo.

    Cada curva é normalizada pelo próprio pico — assim a comparação é de forma e
    posição do pico, não de magnitude absoluta (que não seria comparável entre
    estados de tamanhos muito diferentes).
    """
    todos = []
    for e in estados:
        serie = pd.to_numeric(
            dados_por_estado[e][feat], errors="coerce"
        ).dropna().values.astype(float)
        if len(serie) > 0:
            todos.append(serie)
    # Limites do eixo x pelos percentis (corta caudas extremas).
    todos_concat = np.concatenate(todos) if todos else np.array([0.0, 1.0])
    x_min, x_max = np.percentile(todos_concat, [0.5, 99.5])
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    xs = np.linspace(x_min, x_max, n_pts)

    fig, ax = plt.subplots(figsize=(11, 6))
    for e in estados:
        serie = pd.to_numeric(
            dados_por_estado[e][feat], errors="coerce"
        ).dropna().values.astype(float)
        n = len(serie)
        if n < 2 or np.allclose(serie, serie[0]):
            continue
        try:
            kde  = gaussian_kde(serie)
            dens = kde(xs)
        except Exception:
            continue
        pico = dens.max()
        dens_norm = dens / pico if pico > 0 else dens
        ax.plot(xs, dens_norm, color=cores[e], linewidth=2.2, alpha=0.85,
                label=f"{e} (n={n:,})")

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Densidade (normalizada por estado)", fontsize=10)
    ax.set_title(f"Distribuição de \"{label}\" por estado — Homicídios X85–Y09",
                 fontsize=12, fontweight="bold")
    ax.legend(ncol=3, fontsize=8.5, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if salvar_como:
        plt.savefig(salvar_como, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {salvar_como}")
    plt.close()


def perfil_categorico_feature(feat, dados_por_estado, estados, cores, label,
                              rotulos_categoria, salvar_como=None):
    """Liga, por estado, os pontos de % de homicídios em cada categoria nomeada.

    Substitui o KDE nas variáveis categóricas: suavizar uma curva entre códigos sem
    ordem nem distância real seria enganoso. A linha aqui é só um guia visual entre
    categorias.
    """
    categorias = list(rotulos_categoria.keys())
    labels_x   = [rotulos_categoria[c] for c in categorias]
    x_pos      = np.arange(len(categorias))

    fig, ax = plt.subplots(figsize=(10, 6))
    for e in estados:
        serie = pd.to_numeric(dados_por_estado[e][feat], errors="coerce").dropna()
        n     = len(serie)
        prop  = (serie.value_counts(normalize=True)
                      .reindex(categorias, fill_value=0.0) * 100)
        ax.plot(x_pos, prop.values, color=cores[e], linewidth=2.2, alpha=0.85,
                marker="o", markersize=5, label=f"{e} (n={n:,})")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_x, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("% dos homicídios no estado", fontsize=10)
    ax.set_title(f"Perfil de \"{label}\" por estado — Homicídios X85–Y09",
                 fontsize=12, fontweight="bold")
    ax.legend(ncol=3, fontsize=8.5, loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if salvar_como:
        plt.savefig(salvar_como, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {salvar_como}")
    plt.close()


# ── Um gráfico por feature (KDE se contínua, perfil % se categórica) ──────────
for feat in features_comuns:
    label = LABELS_FEATURES.get(feat, feat)

    if feat in FEATURES_CONTINUAS:
        linha_continua_feature(
            feat, dados_por_estado, estados_com_dados, CORES_ESTADO, label,
            salvar_como=f"{DIR_FIGURAS}linhas_{feat}.png",
        )
    elif feat in FEATURES_CATEGORICAS:
        rotulos = ROTULOS_CATEGORICOS.get(feat)
        if rotulos is None:
            print(f"  Sem rótulos de categoria definidos para {feat} — pulando.")
            continue
        perfil_categorico_feature(
            feat, dados_por_estado, estados_com_dados, CORES_ESTADO, label,
            rotulos, salvar_como=f"{DIR_FIGURAS}perfil_{feat}.png",
        )
    else:
        print(f"  {feat} não está em FEATURES_CONTINUAS nem FEATURES_CATEGORICAS — pulando.")

# ── Resumo tabular: mediana (contínuas) / moda (categóricas) por estado ───────
linhas_resumo = []
for e in estados_com_dados:
    linha = {"Estado": e}
    for feat in FEATURES_CONTINUAS:
        if feat in dados_por_estado[e].columns:
            med = pd.to_numeric(dados_por_estado[e][feat], errors="coerce").median()
            linha[LABELS_FEATURES.get(feat, feat)] = round(med, 1)
    for feat in FEATURES_CATEGORICAS:
        if feat in dados_por_estado[e].columns and feat in ROTULOS_CATEGORICOS:
            serie = pd.to_numeric(dados_por_estado[e][feat], errors="coerce").dropna()
            if len(serie) == 0:
                linha[LABELS_FEATURES.get(feat, feat)] = "-"
                continue
            moda_cod   = serie.mode().iloc[0]
            moda_pct   = (serie == moda_cod).mean() * 100
            moda_label = ROTULOS_CATEGORICOS[feat].get(int(moda_cod), str(moda_cod))
            linha[LABELS_FEATURES.get(feat, feat)] = f"{moda_label} ({moda_pct:.0f}%)"
    linhas_resumo.append(linha)

df_resumo = pd.DataFrame(linhas_resumo).set_index("Estado")
print("\nResumo por estado — mediana (contínuas) / moda (categóricas):")
print(df_resumo.to_string())
df_resumo.to_csv(DIR_RESULTADOS + "resumo_features_por_estado.csv")
print(f"\n  Salvo: {DIR_RESULTADOS}resumo_features_por_estado.csv")

print("\nBloco concluído (linhas sobrepostas: KDE para contínuas, perfil % para categóricas).")
