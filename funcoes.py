import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import hdbscan as _hdb

from config import CID10_HOMICIDIO

# Silencia avisos de terceiros (convergência, divisões por zero etc.) para não
# poluir o log do pipeline.
warnings.filterwarnings("ignore")

# Pacote hdbscan standalone, não o sklearn.cluster.HDBSCAN — este último conflita
# com numpy 2.x na versão usada aqui.
HDBSCAN = _hdb.HDBSCAN

# ── Decodificação dos códigos do SIM ──────────────────────────────────────────
# O SIM grava cada variável como código numérico. Os mapas abaixo traduzem esses
# códigos (e também rótulos textuais, caso o dado já venha decodificado) para os
# valores numéricos usados na via descritiva (preparar_features). Código 0/""/9
# cai em "ignorado".

MAPA_SEXO = {
    "1": 1, "2": 2, "0": 0, "": "0", "M": 1, "F": 2, "I": 0,
    "masculino": 1, "feminino": 2, "ignorado": 0,
}
MAPA_RACA = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "": 0,
    "branca": 1, "preta": 2, "amarela": 3, "parda": 4, "indigena": 5,
}
MAPA_ESTCIV = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "9": 0, "": 0,
    "solteiro": 1, "casado": 2, "viuvo": 3, "separado": 4,
    "união estável": 5, "ignorado": 0,
}
MAPA_ESC = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "9": 0, "0": 0, "": 0,
    "nenhuma": 1, "1 a 3 anos": 2, "4 a 7 anos": 3,
    "8 a 11 anos": 4, "12 anos e mais": 5, "ignorado": 0,
}
MAPA_LOCOCOR = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "9": 0, "": 0,
    "hospital": 1, "outros estabelecimentos de saude": 2,
    "domicílio": 3, "via pública": 4, "outros": 5, "ignorado": 0,
}

# Colunas mostradas na tabela de perfil de cada cluster (perfil_clusters).
COLUNAS_PERFIL = [
    "FAIXA_IDADE", "TURNO_HORA", "CAT_SEXO", "CAT_RACA",
    "CAT_LOCOCOR", "CAT_ESC", "CAT_ESTCIV",
]

# Parâmetros do UMAP (só visualização — ver reduzir_umap/visualizar_umap).
UMAP_N_COMPONENTS = 12
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1


# ── Funções ───────────────────────────────────────────────────────────────────

def dunn_index(X: np.ndarray, labels: np.ndarray,
               max_intra_sample: int = 500) -> float:
    """Índice de Dunn: separação entre clusters ÷ dispersão dentro deles.

    Definido como min(distância entre clusters) / max(diâmetro de um cluster);
    quanto maior, melhor. A parte inter-cluster usa centroides (custa O(K²)); a
    intra-cluster (mais cara) é estimada amostrando até `max_intra_sample` pontos
    por cluster, para não explodir em clusters grandes.
    """
    unique = np.unique(labels)
    centroides = np.array([X[labels == k].mean(axis=0) for k in unique])
    dist_inter = pairwise_distances(centroides)
    np.fill_diagonal(dist_inter, np.inf)
    min_inter = dist_inter.min()

    max_intra = 0.0
    for k in unique:
        pts = X[labels == k]
        if len(pts) > max_intra_sample:
            rng = np.random.default_rng(42)
            pts = pts[rng.choice(len(pts), max_intra_sample, replace=False)]
        if len(pts) > 1:
            max_intra = max(max_intra, pairwise_distances(pts).max())

    return float(min_inter / max_intra) if max_intra > 0 else 0.0


def decodificar_idade(valor):
    """Converte o código de idade do SIM em idade aproximada em anos.

    O 1º dígito indica a unidade (4=anos, 3=meses, 2=dias, 1=horas) e o restante,
    a quantidade. Códigos vazios, malformados ou desconhecidos viram NaN.
    """
    try:
        if pd.isna(valor) or str(valor).strip() in ("", "nan", "None"):
            return np.nan
        s = str(int(float(valor)))
        if len(s) < 2:
            return np.nan
        tipo, quant = int(s[0]), int(s[1:])
        return {4: quant, 3: quant / 12, 2: quant / 365, 1: quant / 8760}.get(tipo, np.nan)
    except Exception:
        return np.nan


def preparar_features(df_estado: pd.DataFrame):
    """Pré-processamento NUMÉRICO (imputação + StandardScaler), via descritiva.

    Mantém cada variável como uma coluna numérica (não faz OHE). É o espaço usado
    só na etapa 04 (distribuições). A clusterização usa preparar_features_ohe, que
    é um espaço diferente — não confundir os dois.
    """
    df = df_estado.copy()
    df["HOMICIDIO"]   = df["CAUSABAS_O"].astype(str).str[:3].isin(CID10_HOMICIDIO).astype(int)
    df["IDADE_NUM"]   = df["IDADE"].apply(decodificar_idade)
    df["HORA_NUM"]    = pd.to_numeric(df["HORAOBITO"].astype(str).str[:2], errors="coerce")
    # Hora fora de 00–23 (ex.: código 99 do SIM = "ignorado") vira NaN, para não
    # distorcer a mediana e o KDE de hora.
    df.loc[(df["HORA_NUM"] < 0) | (df["HORA_NUM"] > 23), "HORA_NUM"] = np.nan
    df["SEXO_NUM"]    = df["SEXO"].astype(str).str.strip().map(MAPA_SEXO)
    df["RACACOR_NUM"] = df["RACACOR"].astype(str).str.strip().map(MAPA_RACA)
    df["ESTCIV_NUM"]  = df["ESTCIV"].astype(str).str.strip().map(MAPA_ESTCIV)
    df["ESC_NUM"]     = df["ESC"].astype(str).str.strip().map(MAPA_ESC)
    df["LOCOCOR_NUM"] = df["LOCOCOR"].astype(str).str.strip().map(MAPA_LOCOCOR)
    df["ANO_NUM"]     = pd.to_numeric(df["ANO"], errors="coerce")

    FEATURES = [
        "IDADE_NUM", "HORA_NUM", "SEXO_NUM", "RACACOR_NUM",
        "ESTCIV_NUM", "ESC_NUM", "LOCOCOR_NUM", "ANO_NUM",
    ]

    # Só entram no modelo as colunas que têm ao menos um valor válido no estado;
    # contínuas são imputadas pela mediana e categóricas pela moda.
    df_mod = df[FEATURES + ["HOMICIDIO"]].copy()
    cont_ok = [f for f in ["IDADE_NUM", "HORA_NUM", "ANO_NUM"]
               if df_mod[f].notna().any()]
    cat_ok  = [f for f in ["SEXO_NUM", "RACACOR_NUM", "ESTCIV_NUM", "ESC_NUM", "LOCOCOR_NUM"]
               if df_mod[f].notna().any()]
    if cont_ok:
        df_mod[cont_ok] = SimpleImputer(strategy="median").fit_transform(df_mod[cont_ok])
    if cat_ok:
        df_mod[cat_ok]  = SimpleImputer(strategy="most_frequent").fit_transform(df_mod[cat_ok])

    features = cont_ok + cat_ok
    X        = df_mod[features].values
    y_true   = df_mod["HOMICIDIO"].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_true, features, df_mod


def categorizar_idade(idade_anos):
    """Agrupa a idade em faixas etárias relevantes para o contexto de homicídio.

    Separa adolescente (12–17) de jovem adulto (18–29), distinção que importa no
    fenômeno (ECA vs. Código Penal). Idade ausente vira "ignorado".
    """
    if pd.isna(idade_anos):
        return "ignorado"
    if idade_anos < 1:
        return "recem_nascido"
    if idade_anos < 12:
        return "crianca"
    if idade_anos < 18:
        return "adolescente"
    if idade_anos < 30:
        return "jovem_adulto"
    if idade_anos < 60:
        return "adulto"
    return "idoso"


def categorizar_hora(hora):
    """Agrupa a hora do óbito em turnos do dia.

    Qualquer hora fora de 00–23 vira "ignorado", inclusive o código 99 do SIM
    (HORAOBITO "9999" → "99"). Tratar 99 como ignorado evita que ele caia em
    "noite" e infle artificialmente esse turno.
    """
    if pd.isna(hora):
        return "ignorado"
    try:
        hora = int(hora)
    except (ValueError, TypeError):
        return "ignorado"
    if hora < 0 or hora > 23:
        return "ignorado"
    if hora < 6:
        return "madrugada"
    if hora < 12:
        return "manha"
    if hora < 18:
        return "tarde"
    return "noite"


def categorizar_sexo(valor):
    """Normaliza o sexo em masculino / feminino / nao_decl."""
    mapa = {
        "1": "masculino", "M": "masculino", "masculino": "masculino",
        "2": "feminino",  "F": "feminino",  "feminino":  "feminino",
        "0": "nao_decl",  "I": "nao_decl",  "ignorado":  "nao_decl",
    }
    return mapa.get(str(valor).strip(), "nao_decl")


def categorizar_raca(valor):
    """Normaliza raça/cor nas cinco categorias do IBGE (+ nao_decl)."""
    mapa = {
        "1": "branca",   "branca":   "branca",
        "2": "preta",    "preta":    "preta",
        "3": "amarela",  "amarela":  "amarela",
        "4": "parda",    "parda":    "parda",
        "5": "indigena", "indigena": "indigena",
        "0": "nao_decl", "":         "nao_decl",
    }
    return mapa.get(str(valor).strip(), "nao_decl")


def categorizar_estciv(valor):
    """Agrupa o estado civil em 3 perfis (+ ignorado).

    Viúvo e separado/divorciado vão juntos em "vinculo_encerrado": demograficamente
    se distinguem do solteiro jovem, que é o perfil mais comum nos homicídios.
    """
    mapa = {
        "1": "solteiro",          "solteiro":      "solteiro",
        "2": "com_vinculo",       "casado":        "com_vinculo",
        "5": "com_vinculo",       "união estável": "com_vinculo",
        "3": "vinculo_encerrado", "viuvo":         "vinculo_encerrado",
        "4": "vinculo_encerrado", "separado":      "vinculo_encerrado",
        "9": "ignorado",          "ignorado":      "ignorado",
        "0": "ignorado",          "":              "ignorado",
    }
    return mapa.get(str(valor).strip(), "ignorado")


def categorizar_esc(valor):
    """Agrupa a escolaridade por nível de ensino (+ ignorado).

    "fundamental" reúne 1–3 e 4–7 anos (fundamental I e II), que no SIM são códigos
    separados.
    """
    mapa = {
        "1": "sem_escolaridade", "nenhuma":        "sem_escolaridade",
        "2": "fundamental",      "1 a 3 anos":     "fundamental",
        "3": "fundamental",      "4 a 7 anos":     "fundamental",
        "4": "medio",            "8 a 11 anos":    "medio",
        "5": "superior",         "12 anos e mais": "superior",
        "9": "ignorado",         "ignorado":       "ignorado",
        "0": "ignorado",         "":               "ignorado",
    }
    return mapa.get(str(valor).strip(), "ignorado")


def categorizar_lococor(valor):
    """Agrupa o local da ocorrência (+ ignorado).

    Hospital e outros estabelecimentos de saúde vão juntos. Via pública fica
    isolada por ser um marcador forte de morte violenta.
    """
    mapa = {
        "1": "estab_saude",   "hospital":                         "estab_saude",
        "2": "estab_saude",   "outros estabelecimentos de saude": "estab_saude",
        "3": "domicilio",     "domicílio":                        "domicilio",
        "4": "local_publico", "via pública":                      "local_publico",
        "5": "outros",        "outros":                           "outros",
        "9": "ignorado",      "ignorado":                         "ignorado",
        "0": "ignorado",      "":                                 "ignorado",
    }
    return mapa.get(str(valor).strip(), "ignorado")


def preparar_features_ohe(df_estado: pd.DataFrame,
                          manter_ignorado_em=("TURNO_HORA",)):
    """Pré-processamento com OHE — este é o espaço usado na clusterização (03, 05–08).

    Supõe que `df_estado` já contém SÓ homicídios confirmados (X85–Y09): o objetivo
    é achar tipologias dentro dos homicídios, não separar homicídio de não-homicídio.
    Faz OHE das nominais (faixa etária, turno, sexo, raça, local), codificação
    ordinal de escolaridade e estado civil, e escala com MaxAbsScaler.

    manter_ignorado_em: colunas cujo "ignorado" vira categoria própria. Por padrão
        só o turno, onde a ausência é alta (RR ~31%, PA ~24%) e carrega informação;
        nas demais nominais a ausência é < 2% e o "ignorado" é descartado, virando a
        categoria de referência implícita.

    Retorna: X_scaled, feature_names, df_modelo.
    """
    df = df_estado.copy()

    # Cada variável vira sua versão categorizada (rótulos legíveis).
    df["FAIXA_IDADE"] = df["IDADE"].apply(decodificar_idade).apply(categorizar_idade)
    df["TURNO_HORA"]  = pd.to_numeric(
        df["HORAOBITO"].astype(str).str[:2], errors="coerce"
    ).apply(categorizar_hora)
    df["CAT_SEXO"]    = df["SEXO"].apply(categorizar_sexo)
    df["CAT_RACA"]    = df["RACACOR"].apply(categorizar_raca)
    df["CAT_LOCOCOR"] = df["LOCOCOR"].apply(categorizar_lococor)
    df["CAT_ESC"]     = df["ESC"].apply(categorizar_esc)
    df["CAT_ESTCIV"]  = df["ESTCIV"].apply(categorizar_estciv)

    # Escolaridade e estado civil entram como ordinais (têm ordem natural), não OHE.
    MAPA_ESC_ORD = {
        "sem_escolaridade": 1, "fundamental": 2,
        "medio": 3,            "superior": 4, "ignorado": 0,
    }
    MAPA_ESTCIV_ORD = {
        "solteiro": 1, "com_vinculo": 2,
        "vinculo_encerrado": 3, "ignorado": 0,
    }
    df["ESC_ORD"]    = df["CAT_ESC"].map(MAPA_ESC_ORD)
    df["ESTCIV_ORD"] = df["CAT_ESTCIV"].map(MAPA_ESTCIV_ORD)

    colunas_ohe = ["FAIXA_IDADE", "TURNO_HORA", "CAT_SEXO", "CAT_RACA", "CAT_LOCOCOR"]
    df_ohe = pd.get_dummies(
        df[colunas_ohe], columns=colunas_ohe, drop_first=False, dtype=float
    )

    # Mantém a coluna "ignorado" só nas variáveis de manter_ignorado_em; nas demais
    # a remove (ausência desprezível → vira a categoria de referência implícita).
    cols_drop_ignorado = []
    for c in df_ohe.columns:
        if "ignorado" not in c:
            continue
        base = c.rsplit("_", 1)[0]
        if base not in manter_ignorado_em:
            cols_drop_ignorado.append(c)
    df_ohe = df_ohe.drop(columns=cols_drop_ignorado, errors="ignore")

    cols_ordinais    = ["ESC_ORD", "ESTCIV_ORD"]
    cols_ohe         = list(df_ohe.columns)
    cols_categoricas = colunas_ohe + ["CAT_ESC", "CAT_ESTCIV"]

    # df_mod guarda tanto as categorias legíveis (para o perfil) quanto as colunas
    # numéricas que de fato vão para o modelo.
    df_mod = pd.concat([
        df[cols_categoricas].reset_index(drop=True),
        df[cols_ordinais].reset_index(drop=True),
        df_ohe.reset_index(drop=True),
    ], axis=1)

    ord_ok = [c for c in cols_ordinais if df_mod[c].notna().any()]
    if ord_ok:
        df_mod[ord_ok] = SimpleImputer(strategy="most_frequent").fit_transform(df_mod[ord_ok])
    df_mod[cols_ohe] = df_mod[cols_ohe].fillna(0)

    # MaxAbsScaler (e não StandardScaler): preserva a esparsidade do OHE, mantendo
    # as colunas binárias em 0/1 e só comprimindo as ordinais para [0, 1].
    features = ord_ok + cols_ohe
    X        = df_mod[features].values.astype(float)
    scaler   = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, features, df_mod


def calcular_metricas_cluster(X, labels, sil_sample=15_000, seed=42):
    """Calcula Silhouette (métrica Hamming), Davies-Bouldin e Dunn de uma partição.

    Pontos de ruído (label −1) são ignorados, para funcionar com DBSCAN/HDBSCAN.
    Retorna NaN em todas as métricas se sobrar menos de 2 clusters (ou < 20 pontos),
    onde as métricas não fazem sentido.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    unique  = np.unique(labels)
    validos = unique[unique >= 0]

    if len(validos) < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Dunn_Index": np.nan}

    mask  = labels >= 0
    X_c   = X[mask]
    lbl_c = labels[mask]

    if len(X_c) < 20:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Dunn_Index": np.nan}

    # Hamming porque as features são quase todas binárias (OHE); a Silhouette é
    # estimada numa amostra (sil_sample) para não custar O(n²) em estados grandes.
    sil  = silhouette_score(
        X_c, lbl_c, metric="hamming", sample_size=min(sil_sample, len(X_c)), random_state=seed
    )
    dbi  = davies_bouldin_score(X_c, lbl_c)
    dunn = dunn_index(X_c, lbl_c)

    return {
        "Silhouette":     round(float(sil),  4),
        "Davies-Bouldin": round(float(dbi),  4),
        "Dunn_Index":     round(float(dunn), 4),
    }


def estabilidade_bootstrap(X: np.ndarray, k: int, n_bootstrap: int = 20,
                            seed: int = 42) -> dict:
    """Mede o quão estável é a partição do K-Means via bootstrap.

    Cada iteração reamostra X com reposição, reclusteriza, realinha os rótulos com
    os da base via linear_sum_assignment (o ARI não pode ser penalizado por mera
    permutação dos rótulos) e calcula o ARI. ARI médio alto ≈ clusters estáveis.
    Retorna média, desvio e a lista de ARIs.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(seed)
    km_base = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels_base = km_base.fit_predict(X)

    scores = []
    for i in range(n_bootstrap):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot = X[idx]
        km_b = KMeans(n_clusters=k, n_init=5, random_state=seed + i)
        labels_boot_raw = km_b.fit_predict(X_boot)
        # Casa cada rótulo do bootstrap com o rótulo correspondente da base
        # (problema de atribuição) antes de comparar.
        labels_base_boot = labels_base[idx]
        cost = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                cost[a, b] = -np.sum((labels_base_boot == a) & (labels_boot_raw == b))
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {col_ind[r]: row_ind[r] for r in range(k)}
        labels_boot_aligned = np.array([mapping[l] for l in labels_boot_raw])
        scores.append(adjusted_rand_score(labels_base_boot, labels_boot_aligned))

    return {"media": float(np.mean(scores)),
            "std":   float(np.std(scores)),
            "valores": scores}


def reduzir_umap(X_scaled, n_components=UMAP_N_COMPONENTS,
                 n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                 seed=42, verbose=True):
    """Projeta X_scaled para n_components dimensões via UMAP.

    Testada como pré-passo da clusterização e DESCARTADA: não melhora as métricas
    em relação ao OHE direto e ainda deixa a curva Silhouette×K não-monotônica.
    Fica aqui só para um eventual reteste — nenhum script atual a chama.
    """
    import umap as umap_lib

    if verbose:
        print(f"  Reduzindo {X_scaled.shape[1]}D → {n_components}D via UMAP...",
              end=" ", flush=True)
    t0 = time.time()
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        n_jobs=1,
    )
    X_reduzido = reducer.fit_transform(X_scaled)
    if verbose:
        print(f"{time.time() - t0:.0f}s")
    return X_reduzido


def visualizar_umap(X_scaled, labels, titulo, estado,
                    df_mod=None, coluna_destaque=None,
                    n_sample=10_000, seed=42, salvar_como=None):
    """Projeta os clusters em 2D via UMAP e salva a figura (só para visualização).

    Painel 1 (sempre): pontos coloridos por cluster, com ruído (−1) em cinza.
    Painel 2 (opcional): os mesmos pontos coloridos pela `coluna_destaque` de
    df_mod — útil para ver se um cluster corresponde a alguma categoria.
    Em estados grandes, amostra n_sample pontos antes de ajustar o UMAP.
    """
    import umap as umap_lib

    n   = len(X_scaled)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)

    X_samp      = X_scaled[idx]
    labels_samp = labels[idx]

    print(f"  Ajustando UMAP em {len(idx):,} pontos...", end=" ", flush=True)
    t0 = time.time()
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        random_state=seed, n_jobs=1,
    )
    emb = reducer.fit_transform(X_samp)
    print(f"{time.time() - t0:.0f}s")

    unique_labels = np.unique(labels_samp)
    n_clust       = len(unique_labels[unique_labels >= 0])
    paleta        = plt.cm.tab10(np.linspace(0, 1, max(n_clust, 2)))

    usa_destaque = df_mod is not None and coluna_destaque is not None
    n_panels     = 2 if usa_destaque else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"UMAP 2D — {titulo} | SIM · {estado} (homicídios confirmados X85–Y09)",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    ax.set_title("Clusters encontrados", fontweight="bold")
    for k in sorted(unique_labels):
        m = labels_samp == k
        if k == -1:
            ax.scatter(emb[m, 0], emb[m, 1], c="lightgray", s=2, alpha=0.35,
                       label=f"Ruído  (n={m.sum():,})", zorder=1)
        else:
            ax.scatter(emb[m, 0], emb[m, 1], c=[paleta[k % 10]], s=6, alpha=0.65,
                       label=f"Cluster {k}  (n={m.sum():,})", zorder=2)
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.3)

    if usa_destaque:
        ax = axes[1]
        ax.set_title(f"Destaque: {coluna_destaque}", fontweight="bold")
        valores    = df_mod[coluna_destaque].values[idx]
        categorias = sorted(pd.unique(valores))
        paleta2    = plt.cm.tab10(np.linspace(0, 1, max(len(categorias), 2)))
        for i, cat in enumerate(categorias):
            m = valores == cat
            ax.scatter(emb[m, 0], emb[m, 1], c=[paleta2[i % 10]], s=4, alpha=0.5,
                       label=f"{cat}  (n={m.sum():,})")
        ax.legend(markerscale=3, fontsize=8, loc="upper right")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if salvar_como:
        plt.savefig(salvar_como, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {salvar_como}")
    plt.show()


def perfil_clusters(df_mod: pd.DataFrame, labels: np.ndarray,
                    colunas: list = None) -> pd.DataFrame:
    """Resume o perfil de cada cluster: tamanho e categoria predominante por variável.

    Uma linha por cluster com N, %, e para cada coluna de `colunas` (default
    COLUNAS_PERFIL) a categoria mais comum, sua porcentagem e o `lift` em relação à
    base global (lift > 1 = categoria sobre-representada no cluster). Ruído (−1)
    aparece como a linha "Ruído".
    """
    if colunas is None:
        colunas = COLUNAS_PERFIL

    df_p = df_mod[colunas].copy()
    df_p["Cluster"] = labels
    total = len(df_p)
    global_props = {col: df_p[col].value_counts(normalize=True) for col in colunas}

    linhas = []
    for cl in sorted(df_p["Cluster"].unique()):
        nome = "Ruído" if cl == -1 else f"Cluster {cl}"
        sub  = df_p[df_p["Cluster"] == cl]
        linha = {
            "Cluster": nome,
            "N":       len(sub),
            "%":       round(len(sub) / total * 100, 1),
        }
        for col in colunas:
            cont = sub[col].value_counts(normalize=True)
            if len(cont) > 0:
                linha[col] = f"{cont.index[0]} ({cont.iloc[0]*100:.0f}%)"
                prop_global = global_props[col].get(cont.index[0], 0)
                linha[f"{col}_lift"] = round(cont.iloc[0] / prop_global, 2) if prop_global > 0 else float("nan")
            else:
                linha[col] = "-"
                linha[f"{col}_lift"] = float("nan")
        linhas.append(linha)

    return pd.DataFrame(linhas).set_index("Cluster")


def rodar_pipeline_estado(
    nome_algoritmo: str,   # nome no cabeçalho do log, ex.: "DBSCAN"
    algo_key: str,         # chave no dict de resultados, ex.: "DBSCAN"
    concluido_msg: str,    # mensagem de encerramento, ex.: "DBSCAN concluído..."
    df_homicidios: pd.DataFrame,
    estados: list,
    clusteriza_fn,         # (df_estado, estado, ctx) -> (df_metricas, melhor_param, labels, perfil)
    resultados: dict,
    pkl_path: str,
    cache: dict = None,    # {estado: (X_scaled, features, df_mod)} de build_cache
) -> dict:
    """Esqueleto comum a DBSCAN, HDBSCAN e Aglomerativo (a parte igual entre eles).

    Para cada estado: imprime o cabeçalho, filtra o df, pula se vazio, obtém o OHE
    (do cache ou recalculando) e delega TODA a lógica específica do algoritmo à
    `clusteriza_fn` — busca de parâmetro, amostragem, figuras, perfil e tabela
    resumo. Guarda resultados[estado][algo_key] no formato padrão e, ao fim,
    persiste o pickle acumulador.

    `clusteriza_fn` recebe (df_estado, estado, ctx) e devolve
    (df_metricas, melhor_param, labels, perfil). O ctx traz t_inicio, X_scaled,
    features, df_mod e n.

    A 05 (K-Means) NÃO usa este esqueleto — tem fluxo próprio (busca de K, heatmap,
    bootstrap e verificação da premissa K=7).
    """
    for estado in estados:
        t_inicio = time.time()
        print("\n" + "█"*70)
        print(f"  {nome_algoritmo} — ESTADO: {estado}")
        print("█"*70)

        df_estado = df_homicidios[df_homicidios["ESTADO"] == estado].copy()
        if df_estado.empty:
            print(f"  Sem dados para {estado}. Pulando.")
            continue

        print(f"  Registros (homicídios confirmados X85–Y09): {len(df_estado):,}")
        if cache is not None and estado in cache:
            X_scaled, features, df_mod = cache[estado]
        else:
            X_scaled, features, df_mod = preparar_features_ohe(df_estado)

        ctx = {"t_inicio": t_inicio, "X_scaled": X_scaled,
               "features": features, "df_mod": df_mod, "n": len(X_scaled)}
        df_metricas, melhor_param, labels, perfil = clusteriza_fn(df_estado, estado, ctx)

        resultados[estado][algo_key] = {
            "df_metricas": df_metricas,
            "melhor_param": melhor_param,
            "labels": labels,
            "perfil": perfil,
        }

    print(f"\n{concluido_msg}")
    with open(pkl_path, "wb") as _f:
        pickle.dump(resultados, _f)
    print(f"  Salvo: {pkl_path}")
    return resultados
