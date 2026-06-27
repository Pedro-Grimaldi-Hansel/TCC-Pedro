import matplotlib
matplotlib.use('Agg')  # backend sem janela — precisa vir antes de importar o pyplot

import os
import runpy

from config import DIR_DADOS, DIR_FIGURAS, DIR_RESULTADOS
from log_setup import iniciar_log

# Garante que as pastas de saída existam antes de qualquer etapa escrever nelas.
os.makedirs(DIR_DADOS, exist_ok=True)
os.makedirs(DIR_FIGURAS, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Um log único para a execução inteira (modo "w": recomeça do zero a cada run).
iniciar_log(DIR_RESULTADOS + "log_resultados.txt")

# Ordem importa: a 01 gera os dados que as demais consomem; 05 cria o cache de
# features que 06–08 reaproveitam; 10 lê os resultados que 05–08 acumularam.
ETAPAS = [
    "01_download.py",
    "02_diagnostico.py",
    "03_hopkins.py",
    "04_distribuicoes.py",
    "05_kmeans.py",
    "06_dbscan.py",
    "07_hdbscan.py",
    "08_aglomerativo.py",
    "09_empacotar.py",
    "10_comparativo.py",
]

# Cada etapa roda em contexto isolado (run_name="__main__"), como se fosse
# chamada direto pelo terminal — por isso a troca de dados entre elas é via disco.
for etapa in ETAPAS:
    print(f"\n{'█'*70}")
    print(f"  RODANDO: {etapa}")
    print(f"{'█'*70}\n")
    runpy.run_path(etapa, run_name="__main__")

print("\n" + "="*70)
print("  PIPELINE CONCLUÍDO")
print("="*70)
