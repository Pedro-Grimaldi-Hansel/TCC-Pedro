# TCC — Tipologias de Homicídio na Amazônia Legal

Pipeline de **clusterização não supervisionada** sobre microdados de mortalidade do
**SIM/DATASUS**. A partir dos óbitos por homicídio confirmado (CID-10 **X85–Y09**) nos
**9 estados da Amazônia Legal** (AC, AM, AP, MA, MT, PA, RO, RR, TO), entre **2013 e 2023**
(~109 mil registros), o projeto busca **tipologias** — subgrupos de perfil de vítima e
circunstância — *dentro* dos homicídios. A clusterização é feita **separadamente por
estado**, para permitir comparação regional.

## Como funciona

O pipeline roda em 10 etapas, uma por arquivo (`01_…` a `10_…`), orquestradas por `main.py`:

| Etapa | O que faz |
|------|-----------|
| 01 | Baixa e converte os arquivos do FTP do DATASUS, unifica e isola os homicídios |
| 02 | Diagnóstico de qualidade do preenchimento (% faltante/ignorado por estado) |
| 03 | Estatística de Hopkins (tendência dos dados a formar clusters) |
| 04 | Distribuições descritivas de cada variável por estado |
| 05–08 | Clusterização por estado: **K-Means, DBSCAN, HDBSCAN e Aglomerativo**, cada um com busca do seu parâmetro, métricas de qualidade, perfis e visualização UMAP 2D |
| 09 | Empacota figuras e log em ZIP |
| 10 | Compara a concordância entre algoritmos (ARI) |

As features vêm de One-Hot Encoding das variáveis nominais (faixa etária, turno, sexo,
raça, local de ocorrência) e codificação ordinal de escolaridade e estado civil. A
qualidade dos clusters é medida por Silhouette, Davies-Bouldin e Dunn. O UMAP é usado
apenas para visualização 2D, nunca como espaço de clusterização.

## Como rodar

```bash
# 1. Ambiente virtual
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# 2. Dependências
pip install -r requirements.txt

# 3. Pipeline completo (01 → 10), com log único
python main.py
```

A etapa 01 baixa os dados do FTP do DATASUS (precisa de internet) e pode demorar. As etapas
seguintes leem os dados já salvos em `dados/`, então podem ser rodadas isoladamente —
por exemplo `python 05_kmeans.py` — desde que a 01 já tenha rodado uma vez.

## Saídas

Tudo é gerado localmente e fica fora do versionamento (ver `.gitignore`):

- `dados/` — `homicidios.parquet` e CSVs intermediários
- `figuras/` — gráficos por estado e algoritmo (PNG)
- `resultados/` — `log_resultados.txt`, métricas em CSV e `resultados.pkl`

## Fonte dos dados

SIM/DATASUS — FTP `ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/`
Anos: 2013–2023 · Estados: AC, AM, AP, MA, MT, PA, RO, RR, TO

## Stack

Python 3.12 · pandas · scikit-learn · hdbscan · umap-learn · scipy · matplotlib · seaborn
(versões completas em `requirements.txt`).
