import os

import numpy as np
import pandas as pd

from config import DIR_DADOS, DIR_RESULTADOS, ESTADOS
from funcoes import decodificar_idade

# ── Carrega dados ─────────────────────────────────────────────────────────────
os.makedirs(DIR_RESULTADOS, exist_ok=True)
df_homicidios = pd.read_parquet(DIR_DADOS + "homicidios.parquet")

# ── O que conta como "ignorado" em cada variável ──────────────────────────────
# Códigos que significam "preenchido, mas sem informação" (distinto de faltante).
CODIGOS_IGNORADO = {
    "SEXO":    {"0", "I", "ignorado", "nao_decl"},
    "RACACOR": set(),                       # raça não tem código de ignorado; só ausência
    "ESTCIV":  {"9", "ignorado"},
    "ESC":     {"9", "ignorado"},
    "LOCOCOR": {"9", "ignorado"},
}

# Strings que representam ausência de valor.
VAZIOS = {"", " ", "nan", "none", "None", "NaN", "<NA>"}

COLS_CATEGORICAS = ["SEXO", "RACACOR", "ESTCIV", "ESC", "LOCOCOR"]


# ── Funções locais ────────────────────────────────────────────────────────────

def pct_faltante_ignorado(serie: pd.Series, codigos_ignorado: set):
    """Devolve (%_faltante, %_ignorado) de uma coluna categórica crua.

    Faltante = valor vazio/nulo; ignorado = código de "ignorado". Os dois são
    contados sem sobreposição (o ignorado só entre os não-faltantes).
    """
    s = serie.astype(str).str.strip()
    n = len(s)
    if n == 0:
        return 0.0, 0.0
    falt = s.str.lower().isin({v.lower() for v in VAZIOS}) | serie.isna()
    pct_falt = falt.mean() * 100
    ign = (~falt) & s.isin(codigos_ignorado)
    pct_ign = ign.mean() * 100
    return round(pct_falt, 1), round(pct_ign, 1)


def pct_idade_invalida(df):
    """% de IDADE que decodificar_idade não consegue interpretar (vira NaN)."""
    dec = df["IDADE"].apply(decodificar_idade)
    return round(dec.isna().mean() * 100, 1)


def pct_hora_invalida(df):
    """% de HORAOBITO que não cai numa hora válida 00–23 (inclui o código 99)."""
    h = pd.to_numeric(df["HORAOBITO"].astype(str).str[:2], errors="coerce")
    invalida = h.isna() | (h < 0) | (h > 23)
    return round(invalida.mean() * 100, 1)


# ── Monta a tabela: uma linha por estado, % sem info por variável ─────────────
linhas = []
for ESTADO in ESTADOS:
    df_e = df_homicidios[df_homicidios["ESTADO"] == ESTADO]
    if df_e.empty:
        continue

    linha = {"Estado": ESTADO, "N": len(df_e)}

    for col in COLS_CATEGORICAS:
        if col in df_e.columns:
            pf, pi = pct_faltante_ignorado(df_e[col], CODIGOS_IGNORADO.get(col, set()))
            linha[f"{col}_falt%"]  = pf
            linha[f"{col}_ign%"]   = pi
            linha[f"{col}_total%"] = round(pf + pi, 1)

    if "IDADE" in df_e.columns:
        linha["IDADE_inval%"] = pct_idade_invalida(df_e)
    if "HORAOBITO" in df_e.columns:
        linha["HORA_inval%"] = pct_hora_invalida(df_e)

    linhas.append(linha)

df_diag = pd.DataFrame(linhas).set_index("Estado")

# ── Visão 1: total "sem informação útil" por variável (faltante + ignorado) ───
cols_total = [c for c in df_diag.columns if c.endswith("_total%")] \
           + [c for c in df_diag.columns if c.endswith("_inval%")]
cols_total = ["N"] + cols_total

print("=" * 78)
print("  DIAGNÓSTICO — % SEM INFORMAÇÃO ÚTIL (faltante + ignorado) por estado")
print("=" * 78)
print(df_diag[cols_total].to_string())
print()

# ── Visão 2: alertas onde a ausência passa de 15% (impacta a imputação) ───────
print("=" * 78)
print("  ALERTAS (variáveis com > 15% sem informação útil em algum estado)")
print("=" * 78)
LIMIAR_ALERTA = 15.0
algum_alerta = False
for col in cols_total:
    if col == "N":
        continue
    for est in df_diag.index:
        val = df_diag.loc[est, col]
        if pd.notna(val) and val > LIMIAR_ALERTA:
            sev = "⚠⚠ ALTO" if val > 30 else "⚠ moderado"
            print(f"  {sev:12s} | {est} · "
                  f"{col.replace('_total%','').replace('_inval%','')}: {val:.1f}%")
            algum_alerta = True
if not algum_alerta:
    print("  ✓ Nenhuma variável passou de 15% em nenhum estado — imputação de "
          "baixo impacto.")
print("=" * 78)

# ── Visão 3: média entre estados, com barra para leitura rápida ───────────────
print("\n  Média entre estados (% sem info útil por variável):")
for col in cols_total:
    if col == "N":
        continue
    media = df_diag[col].mean()
    nome  = col.replace("_total%", "").replace("_inval%", " (inválido)")
    barra = "█" * int(media / 2)
    print(f"    {nome:22s} {media:5.1f}%  {barra}")

# ── Salva o CSV de diagnóstico (insumo da seção de Limitações do TCC) ─────────
df_diag.to_csv(DIR_RESULTADOS + "diagnostico_qualidade_preenchimento.csv")
print(f"\n  Salvo: {DIR_RESULTADOS}diagnostico_qualidade_preenchimento.csv")
print("\n  Use estes números na seção de LIMITAÇÕES do TCC: declare o % de "
      "imputação\n  por variável e ressalve conclusões sobre variáveis com "
      "alta ausência.")
