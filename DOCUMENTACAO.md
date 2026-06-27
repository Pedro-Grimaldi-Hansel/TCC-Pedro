# Documentação — Pipeline de Tipologias de Homicídio na Amazônia Legal

Documentação estruturada em três níveis dos 15 arquivos `.py` do projeto. Gerada a partir
da leitura direta do código (estado atual). Pensada para acompanhar os resultados (figuras,
CSVs, log) em uma avaliação posterior do projeto.

---

## Nível 1 — Visão geral do projeto

Este projeto é um **pipeline de clusterização não supervisionada** aplicado a microdados de
mortalidade do **SIM/DATASUS**. A partir dos arquivos brutos de óbito (DBC do FTP do DATASUS),
ele isola os **homicídios confirmados** (CID-10 X85–Y09) dos **9 estados da Amazônia Legal**
(AC, AM, AP, MA, MT, PA, RO, RR, TO) no período **2013–2023** (~109 mil registros) e busca,
**dentro** dos homicídios, **tipologias** (subgrupos de perfil de vítima/circunstância) por
meio de clusterização — feita **separadamente por estado** para permitir comparação regional.

O fluxo, do início ao fim, é: (1) **download e conversão** dos arquivos DBC→DBF do FTP,
unificação num único DataFrame e geração de `homicidios.parquet`; (2) **diagnóstico de
qualidade** do preenchimento das variáveis (% faltante/ignorado por estado); (3) teste da
**Estatística de Hopkins** para verificar se os dados têm tendência real a formar clusters;
(4) **distribuições descritivas** de cada variável por estado (KDE para contínuas, perfil %
para categóricas); (5–8) quatro **algoritmos de clusterização** — K-Means, DBSCAN, HDBSCAN e
Aglomerativo — rodados por estado, cada um com busca do seu parâmetro principal, métricas de
qualidade, perfis de cluster e visualização UMAP 2D; (9) **empacotamento** das figuras e do
log em ZIP; e (10) comparação **ARI** entre os algoritmos para medir concordância.

Metodologicamente, as features são preparadas com **One-Hot Encoding** das variáveis nominais
(faixa etária, turno, sexo, raça, local de ocorrência) mais codificação ordinal de
escolaridade e estado civil, escaladas com **MaxAbsScaler** (não StandardScaler) para
preservar a esparsidade do OHE. A qualidade dos clusters é medida por **Silhouette (métrica
Hamming)**, **Davies-Bouldin** e **Dunn Index**. O UMAP é usado **apenas para visualização
2D**, nunca como espaço de clusterização (a clusterização ocorre no espaço OHE direto). Os
artefatos pesados são persistidos em disco (`.parquet`, `.pkl`) porque cada etapa roda num
processo separado via `runpy`.

---

## Nível 2 — Mapa de arquivos

> Convenção de paths (de `config.py`): `DIR_DADOS = ./dados/`, `DIR_FIGURAS = ./figuras/`,
> `DIR_RESULTADOS = ./resultados/`, `DIRETORIO_LOCAL = ./dados_sim/` (DBC/DBF brutos).
> O cache de features fica em `./features_cache.pkl` (raiz).

| Arquivo | Responsabilidade (1 frase) | Inputs | Outputs | Dependências internas |
|---|---|---|---|---|
| **config.py** | Centraliza constantes do projeto (estados, cores, CID-10 de homicídio, paths, anos). | nada | nada | nenhuma |
| **funcoes.py** | Biblioteca de funções: feature engineering, métricas, perfil de clusters, UMAP e scaffolding dos algoritmos. | nada (em import) | nada (em import) | `config` |
| **log_setup.py** | Tee de stdout para arquivo + console (com UTF-8 e fallback de encoding); inicia o log. | nada | `log_resultados.txt` (escrito pelo chamador) | nenhuma |
| **_cache_features.py** | Cache em disco do OHE por estado, reaproveitado entre 05–08. | `features_cache.pkl` (se existir) | `features_cache.pkl` | `funcoes`, `config` |
| **01_download.py** | Baixa/converte DBC→DBF do FTP DATASUS, unifica, filtra colunas e isola homicídios. | FTP DATASUS; DBF em `./dados_sim/` | `dados/mortalidade_unificada.csv`, `dados/filtrado.csv`, `dados/filtrado.parquet`, `dados/homicidios.parquet`; remove `features_cache.pkl` | `config` |
| **02_diagnostico.py** | Tabela de % sem informação útil (faltante + ignorado) por variável e estado. | `dados/homicidios.parquet` | `resultados/diagnostico_qualidade_preenchimento.csv` | `config`, `funcoes` |
| **03_hopkins.py** | Estatística de Hopkins por estado (tendência de clustering) + gráfico 3×3. | `dados/homicidios.parquet` | `figuras/hopkins_9estados.png` | `config`, `funcoes` |
| **04_distribuicoes.py** | Distribuições por estado: KDE (contínuas) e perfil % (categóricas) + resumo. | `dados/homicidios.parquet` | `figuras/linhas_*.png`, `figuras/perfil_*.png`, `resultados/resumo_features_por_estado.csv` | `config`, `funcoes` |
| **05_kmeans.py** | K-Means por estado (K=2..10), métricas, perfil, UMAP, bootstrap; **constrói o cache de features**. | `dados/homicidios.parquet`, `features_cache.pkl`, `resultados/resultados.pkl` (se existir) | `features_cache.pkl`, `figuras/kmeans_*_<UF>.png`, `resultados/resultados.pkl` (chave `KMeans`) | `config`, `funcoes`, `_cache_features` |
| **06_dbscan.py** | DBSCAN por estado, com busca de `eps` via k-distância. | `dados/homicidios.parquet`, `features_cache.pkl`, `resultados/resultados.pkl` | `figuras/dbscan_*_<UF>.png`, `resultados/resultados.pkl` (chave `DBSCAN`) | `config`, `funcoes`, `_cache_features` |
| **07_hdbscan.py** | HDBSCAN por estado, varrendo `cluster_selection_epsilon`. | `dados/homicidios.parquet`, `features_cache.pkl`, `resultados/resultados.pkl` | `figuras/hdbscan_*_<UF>.png`, `resultados/resultados.pkl` (chave `HDBSCAN`) | `config`, `funcoes`, `_cache_features` |
| **08_aglomerativo.py** | Aglomerativo hierárquico por estado (ward/average/complete), dendrograma, métricas. | `dados/homicidios.parquet`, `features_cache.pkl`, `resultados/resultados.pkl` | `figuras/aglomerativo_*_<UF>.png`, `resultados/resultados.pkl` (chave `Aglomerativo`) | `config`, `funcoes`, `_cache_features` |
| **09_empacotar.py** | Monta ZIP(s) das figuras essenciais (+ completo opcional) e do log; abre a pasta. | `figuras/*.png`, `resultados/log_resultados.txt` | `resultados/essencial_<ts>.zip` (e `completo_<ts>.zip` se ligado) | `config` |
| **10_comparativo.py** | ARI entre pares de algoritmos por estado. | `resultados/resultados.pkl` | `resultados/ari_entre_algoritmos.csv` | `config` |
| **main.py** | Orquestrador: roda 01→10 em sequência via `runpy`, com log único. | scripts 01–10 | `resultados/log_resultados.txt` (+ tudo que as etapas geram) | `config`, `log_setup` |

---

## Nível 3 — Funções e correlações

### `config.py` — só constantes (não define funções)

| Constante | Conteúdo | Quem usa |
|---|---|---|
| `ESTADOS` | Lista dos 9 estados da Amazônia Legal. | 01–10, `_cache_features` |
| `CORES_ESTADO` | Mapa estado → cor hex (gráficos comparativos). | 03, 04 |
| `COR_BASE` | Cor base (`#6366f1`) para gráficos. | 05, 06, 08 |
| `CID10_HOMICIDIO` | Códigos X85–Y09 que definem homicídio confirmado. | 01, `funcoes.preparar_features` |
| `DIRETORIO_REMOTO`, `DIRETORIO_LOCAL`, `ANOS` | Origem FTP, destino local DBC/DBF, anos 2013–2023. | 01 |
| `DIR_DADOS`, `DIR_FIGURAS`, `DIR_RESULTADOS` | Paths de saída (`./dados/`, `./figuras/`, `./resultados/`). | 01–10, `main` |

---

### `funcoes.py`

Constantes de módulo: `HDBSCAN` (alias do pacote standalone), `MAPA_SEXO/RACA/ESTCIV/ESC/LOCOCOR`
(decodificação numérica), `COLUNAS_PERFIL` (colunas default do perfil), `UMAP_N_COMPONENTS/
N_NEIGHBORS/MIN_DIST`.

| Função / Classe | Assinatura | O que faz | Quem chama |
|---|---|---|---|
| `dunn_index` | `dunn_index(X, labels, max_intra_sample=500) -> float` | Dunn Index = mín. distância inter-cluster (via centroides) ÷ máx. diâmetro intra-cluster (amostrado). Maior = melhor. | `calcular_metricas_cluster` (interno a `funcoes`) |
| `decodificar_idade` | `decodificar_idade(valor)` | Decodifica o código de idade do SIM (1º dígito = unidade, resto = quantidade) para idade em anos. | `preparar_features`, `preparar_features_ohe`, `02_diagnostico` |
| `preparar_features` | `preparar_features(df_estado) -> (X_scaled, y_true, features, df_mod)` | Pré-processamento **numérico** (imputação + **StandardScaler**), com flag `HOMICIDIO`. Usado na via descritiva. | `04_distribuicoes` |
| `categorizar_idade` | `categorizar_idade(idade_anos) -> str` | Agrupa idade em faixas (recém-nascido…idoso); separa adolescente de jovem adulto. | `preparar_features_ohe` |
| `categorizar_hora` | `categorizar_hora(hora) -> str` | Agrupa hora do óbito em turnos; trata código 99/inválidos como "ignorado". | `preparar_features_ohe` |
| `categorizar_sexo` | `categorizar_sexo(valor) -> str` | Normaliza sexo (masculino/feminino/nao_decl). | `preparar_features_ohe` |
| `categorizar_raca` | `categorizar_raca(valor) -> str` | Normaliza raça/cor. | `preparar_features_ohe` |
| `categorizar_estciv` | `categorizar_estciv(valor) -> str` | Estado civil agrupado (solteiro/com_vinculo/vinculo_encerrado/ignorado). | `preparar_features_ohe` |
| `categorizar_esc` | `categorizar_esc(valor) -> str` | Escolaridade por nível (sem/fundamental/medio/superior/ignorado). | `preparar_features_ohe` |
| `categorizar_lococor` | `categorizar_lococor(valor) -> str` | Local de ocorrência (estab_saude/domicilio/local_publico/outros/ignorado). | `preparar_features_ohe` |
| `preparar_features_ohe` | `preparar_features_ohe(df_estado, manter_ignorado_em=("TURNO_HORA",)) -> (X_scaled, features, df_mod)` | Pré-processamento **OHE** das nominais + ordinais (ESC/ESTCIV) + **MaxAbsScaler**; mantém "ignorado" só no turno. Espaço usado na clusterização. | `03_hopkins`, `_cache_features.build_cache`, `rodar_pipeline_estado` (fallback) |
| `calcular_metricas_cluster` | `calcular_metricas_cluster(X, labels, sil_sample=15000, seed=42) -> dict` | Calcula Silhouette (**Hamming**), Davies-Bouldin e Dunn; ignora ruído (−1); NaN se <2 clusters. | 05, 06, 07, 08 |
| `estabilidade_bootstrap` | `estabilidade_bootstrap(X, k, n_bootstrap=20, seed=42) -> dict` | Estabilidade do K-Means via bootstrap: reamostra, realinha rótulos (hungaro) e calcula ARI; retorna média/std/valores. | 05 (estados < 6000 registros) |
| `reduzir_umap` | `reduzir_umap(X_scaled, n_components=12, ...) -> X_reduzido` | Projeta para n_components via UMAP. **Descartada do pipeline padrão**, mantida para reteste. | nenhum (não usado nos scripts atuais) |
| `visualizar_umap` | `visualizar_umap(X_scaled, labels, titulo, estado, df_mod=None, coluna_destaque=None, n_sample=10000, ...)` | UMAP 2D: painel de clusters (+ painel de destaque opcional); salva PNG. | 05, 06, 07, 08 |
| `perfil_clusters` | `perfil_clusters(df_mod, labels, colunas=None) -> DataFrame` | Por cluster: N, %, categoria predominante por coluna e **lift** vs. base global. Ruído como linha "Ruído". | 05, 06, 07, 08 |
| `rodar_pipeline_estado` | `rodar_pipeline_estado(nome_algoritmo, algo_key, concluido_msg, df_homicidios, estados, clusteriza_fn, resultados, pkl_path, cache=None) -> dict` | Scaffolding comum aos algoritmos uniformes: itera estados, prepara/reusa features e delega à `clusteriza_fn`; grava `resultados[estado][algo_key]` e persiste o pickle. | 06, 07, 08 |

> Nota: 05 **não** usa `rodar_pipeline_estado` (tem loop próprio, pois faz busca de K,
> heatmap, bootstrap e verificação consolidada da premissa K=7). 06/07/08 usam o scaffolding.

---

### `log_setup.py`

| Função / Classe | Assinatura | O que faz | Quem chama |
|---|---|---|---|
| `_Tee` | `class _Tee(*streams)` com `write`/`flush` | Stream que escreve em vários destinos (console + arquivo). No `write`, captura `UnicodeEncodeError` e troca caracteres não representáveis (fallback p/ consoles em cp1252) em vez de derrubar o pipeline. | `iniciar_log` |
| `iniciar_log` | `iniciar_log(caminho="log_resultados.txt")` | Reconfigura `stdout`/`stderr` para **UTF-8** (para acentos e `█` saírem certos no Windows), abre o log em modo `"w"` e redireciona `sys.stdout` via `_Tee`. | `main.py` (chamado **uma vez**) |

---

### `_cache_features.py`

Constante de módulo: `_CACHE_PATH = "features_cache.pkl"`.

| Função | Assinatura | O que faz | Quem chama |
|---|---|---|---|
| `build_cache` | `build_cache(df_homicidios) -> dict` | Carrega o cache OHE do disco se existir; senão constrói `{estado: (X_scaled, features, df_mod)}` rodando `preparar_features_ohe` 1× por estado e persiste em `features_cache.pkl`. | 05, 06, 07, 08 |

---

### Scripts de execução — fluxo interno e dependências de ordem

#### `01_download.py`
- **Carrega**: lista o FTP DATASUS; baixa DBC e converte para DBF (retry por arquivo + retomada).
- **Processa**: une todos os DBF (colunas categóricas como `category` para economizar RAM),
  seleciona colunas (`COLUNAS_SELECIONADAS`), libera o DataFrame intermediário (`del` + `gc`),
  e isola homicídios pelo CID-10 (`CID10_HOMICIDIO`).
- **Salva**: `dados/mortalidade_unificada.csv`, `dados/filtrado.csv`, `dados/filtrado.parquet`,
  `dados/homicidios.parquet`. **Remove `features_cache.pkl`** (invalida o cache pois há dados novos).
- **Ordem**: deve rodar **primeiro** — `homicidios.parquet` é consumido por 02–08.

#### `02_diagnostico.py`
- **Carrega**: `dados/homicidios.parquet`.
- **Processa**: por estado e variável categórica, calcula % faltante + % ignorado (e %
  inválido de idade/hora); emite alertas onde passa de 15%/30%; média entre estados.
- **Salva**: `resultados/diagnostico_qualidade_preenchimento.csv` (para a seção de Limitações do TCC).

#### `03_hopkins.py`
- **Carrega**: `dados/homicidios.parquet`.
- **Processa**: por estado, gera features OHE (`preparar_features_ohe`) e calcula a
  Estatística de Hopkins (amostra reduzida em estados pequenos como RR/AP/AC).
- **Salva**: `figuras/hopkins_9estados.png` (grade 3×3) + resumo no log.

#### `04_distribuicoes.py`
- **Carrega**: `dados/homicidios.parquet`.
- **Processa**: usa `preparar_features` (**numérico/StandardScaler**, diferente do OHE);
  por feature, gera KDE sobrepostas (contínuas) ou perfil % por categoria (categóricas).
- **Salva**: `figuras/linhas_<feat>.png`, `figuras/perfil_<feat>.png`,
  `resultados/resumo_features_por_estado.csv`.

#### `05_kmeans.py`
- **Carrega**: `dados/homicidios.parquet`; **constrói/lê o cache** via `build_cache`;
  carrega `resultados/resultados.pkl` (ou inicializa).
- **Processa**: por estado, roda K-Means para K=2..10, calcula métricas, detecta cotovelo e
  K ótimo por Silhouette/DBI/Dunn; **adota o K do pico de Dunn**; gera bootstrap em estados
  pequenos (< 6000 registros); ao fim, **verifica a premissa global K=7** (`K_FINAL`) contra
  o pico de Dunn de cada estado.
- **Salva**: `figuras/kmeans_metricas_<UF>.png`, `kmeans_heatmap_<UF>.png`,
  `kmeans_umap_<UF>.png`; `resultados/resultados.pkl` (chave `KMeans`). Também grava `features_cache.pkl`.
- **Ordem**: deve rodar **antes de 06–08** para criar `features_cache.pkl` (06/07/08 o reconstroem
  se ausente, mas rodar 05 primeiro evita recomputo).

#### `06_dbscan.py`, `07_hdbscan.py`, `08_aglomerativo.py`
- **Carrega**: `dados/homicidios.parquet`, `features_cache.pkl` (via `build_cache`),
  `resultados/resultados.pkl`.
- **Processa** (cada um define uma `clusteriza_fn` e delega ao scaffolding `rodar_pipeline_estado`):
  - **06**: amostra se n > 40k; escolhe `eps` pelo joelho da curva k-distância; varre candidatos e
    escolhe o de maior Silhouette com 2..12 clusters e ruído < 50%.
  - **07**: `min_cluster_size` ~ 1% de n; varre `cluster_selection_epsilon`; mesmo critério de seleção.
  - **08**: amostra (matriz O(n²), até 8k); testa linkage ward/average/complete × K=2..10; exige
    menor cluster ≥ 2%; escolhe melhor Silhouette; gera dendrograma truncado.
- **Salva**: PNGs `<algo>_*_<UF>.png`; acrescenta a chave do algoritmo em `resultados/resultados.pkl`.
- **Ordem**: o `resultados.pkl` é **acumulador** — cada etapa carrega, adiciona sua chave e regrava.

#### `09_empacotar.py`
- **Carrega**: lista `figuras/*.png` e o log.
- **Processa**: monta o conjunto **essencial** (hopkins/linhas/perfil + 1 UMAP por algoritmo,
  pelo estado de maior volume). ZIP completo opcional (`GERAR_ZIP_COMPLETO = False`).
- **Salva**: `resultados/essencial_<timestamp>.zip` (e abre a pasta no explorador).

#### `10_comparativo.py`
- **Carrega**: `resultados/resultados.pkl` (via `DIR_RESULTADOS`).
- **Processa**: para cada estado, calcula ARI entre todos os pares de algoritmos (só pontos
  não-ruído de ambos, com ≥ 50 pontos).
- **Salva**: `resultados/ari_entre_algoritmos.csv` + resumo por par no console.
- **Ordem**: roda **depois de 05–08** (e está incluído no `main.py`, após 09).

#### `main.py`
- Cria os diretórios de saída, chama `iniciar_log` (modo `"w"`, uma vez) e roda
  **01→10** em sequência via `runpy.run_path(..., run_name="__main__")` (cada etapa em
  contexto isolado — daí a persistência em disco entre elas).

---

## Dependências de ordem e observações importantes

1. **`01` é pré-requisito de tudo**: gera `dados/homicidios.parquet` (consumido por 02–08) e
   invalida `features_cache.pkl`.
2. **`05` cria o cache de features** (`features_cache.pkl`) que **06/07/08 reaproveitam**. Como
   `build_cache` reconstrói o cache se ele estiver ausente, 06–08 funcionam isolados, mas a
   ordem 05→06→07→08 evita recomputar o OHE.
3. **`resultados.pkl` é acumulador** entre 05–08: cada etapa carrega o pickle existente,
   adiciona a chave do seu algoritmo e regrava. `10` lê esse pickle para o ARI.
4. **Path do `resultados.pkl` unificado**: 05–08 e `10_comparativo.py` usam todos
   `DIR_RESULTADOS` → `./resultados/`. `10` lê `resultados/resultados.pkl` e grava
   `resultados/ari_entre_algoritmos.csv`, no mesmo diretório de 05–08.
5. **`10_comparativo.py` incluído no `main.py`**: a lista `ETAPAS` vai de 01 a **10**, então o
   comparativo ARI roda automaticamente após 09 no pipeline orquestrado.
6. **Dois pré-processamentos distintos**: `04_distribuicoes.py` usa `preparar_features`
   (numérico, **StandardScaler**) só para descritiva; a clusterização (03, 05–08) usa
   `preparar_features_ohe` (OHE, **MaxAbsScaler**). Não confundir os espaços de features.
7. **UMAP é só visualização** — a clusterização ocorre no espaço OHE direto. `reduzir_umap`
   existe em `funcoes.py` mas não é chamada por nenhum script atual.
8. **Log em UTF-8 com fallback**: `iniciar_log` força UTF-8 em `stdout`/`stderr` e o `_Tee`
   tem fallback para `UnicodeEncodeError`, garantindo que acentos e `█` saiam corretos mesmo
   em console Windows (cp1252) sem interromper a execução.
