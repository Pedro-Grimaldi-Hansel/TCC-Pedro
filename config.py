# Constantes compartilhadas por todo o pipeline. Sem lógica — só valores fixos
# que vários scripts importam, para não duplicar (estados, cores, CID-10, paths).

# ── Estados da Amazônia Legal (recorte do estudo) ─────────────────────────────
ESTADOS = ["AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"]

# Cor fixa por estado, para que o mesmo estado tenha sempre a mesma cor nos
# gráficos comparativos.
CORES_ESTADO = {
    "AC": "#1f77b4",
    "AM": "#ff7f0e",
    "AP": "#2ca02c",
    "MA": "#d62728",
    "MT": "#9467bd",
    "PA": "#8c564b",
    "RO": "#e377c2",
    "RR": "#17becf",
    "TO": "#bcbd22",
}

# Cor padrão dos gráficos de um estado só (sem comparação entre estados).
COR_BASE = "#6366f1"

# ── Definição de homicídio ────────────────────────────────────────────────────
# Agressões (X85–Y09) na CID-10. É o que separa "homicídio confirmado" do resto
# dos óbitos — usado no filtro da etapa 01 e no preparo das features.
CID10_HOMICIDIO = [
    "X85", "X86", "X87", "X88", "X89", "X90", "X91", "X92", "X93", "X94",
    "X95", "X96", "X97", "X98", "X99", "Y00", "Y01", "Y02", "Y03", "Y04",
    "Y05", "Y06", "Y07", "Y08", "Y09",
]

# ── Origem dos dados (FTP do DATASUS) e janela temporal ───────────────────────
DIRETORIO_REMOTO = "/dissemin/publicos/SIM/CID10/DORES/"
DIRETORIO_LOCAL  = "./dados_sim/"
ANOS             = ["2013", "2014", "2015", "2016", "2017", "2018",
                    "2019", "2020", "2021", "2022", "2023"]

# ── Pastas de saída ───────────────────────────────────────────────────────────
# Todo I/O do pipeline passa por estes paths (nunca a raiz do projeto).
DIR_DADOS      = "./dados/"
DIR_FIGURAS    = "./figuras/"
DIR_RESULTADOS = "./resultados/"
