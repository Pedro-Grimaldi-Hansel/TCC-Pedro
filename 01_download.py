import ftplib
import gc
import os
import time

import pandas as pd
import datasus_dbc
from dbfread import DBF

from config import (ANOS, CID10_HOMICIDIO, DIR_DADOS,
                    DIRETORIO_LOCAL, DIRETORIO_REMOTO, ESTADOS)

# Colunas guardadas como `category` para economizar RAM (são poucas categorias
# repetidas em milhões de linhas).
COLUNAS_CATEGORICAS = ["SEXO", "RACACOR", "ESTCIV", "ESC", "LOCOCOR", "GRAVIDEZ", "ESTADO"]

# Subconjunto de colunas do SIM que o estudo usa (o DBF tem muito mais).
COLUNAS_SELECIONADAS = [
    "DTOBITO", "HORAOBITO", "DTNASC", "IDADE", "SEXO", "RACACOR", "ESTCIV", "ESC",
    "LOCOCOR", "GRAVIDEZ", "CAUSABAS_O", "ANO", "ESTADO",
]


def baixar_e_converter_arquivos(diretorio_remoto, diretorio_local, anos,
                                estados, max_tentativas=3,
                                espera_entre_tentativas=5):
    """Baixa do FTP do DATASUS e converte DBC→DBF, de forma resiliente.

    Tenta cada arquivo até `max_tentativas` vezes e retoma de onde parou (pula os
    DBF que já existem), então pode ser rodada de novo sem rebaixar tudo. Devolve a
    lista de arquivos que falharam (vazia = tudo certo).
    """
    if not os.path.exists(diretorio_local):
        os.makedirs(diretorio_local)

    print("  Conectando ao FTP do DATASUS para listar arquivos...")
    ftp = ftplib.FTP("ftp.datasus.gov.br")
    ftp.login()
    ftp.cwd(diretorio_remoto)
    arquivos_disponiveis = ftp.nlst()
    ftp.quit()

    # Mantém só os arquivos dos estados e anos do estudo (padrão "...UFAAAA.dbc").
    arquivos_alvo = sorted(set(
        arq for arq in arquivos_disponiveis
        for estado in estados
        for ano in anos
        if arq.endswith(f"{estado}{ano}.dbc")
    ))

    esperado = len(anos) * len(estados)
    print(f"  {len(arquivos_alvo)} arquivos identificados no servidor "
          f"(esperado: {esperado})")

    if len(arquivos_alvo) != esperado:
        faltando_no_servidor = (
            {f"{e}{a}" for e in estados for a in anos}
            - {arq[2:8] for arq in arquivos_alvo}
        )
        print(f"  ⚠ O servidor não tem todos os arquivos esperados. "
              f"Faltando no FTP: {sorted(faltando_no_servidor)}")

    falhas, sucesso = [], 0

    for arquivo in arquivos_alvo:
        caminho_dbc = os.path.join(diretorio_local, arquivo)
        caminho_dbf = caminho_dbc.replace(".dbc", ".dbf")

        # Retomada: se o DBF já existe, esse arquivo já foi processado antes.
        if os.path.exists(caminho_dbf):
            sucesso += 1
            continue

        ok = False
        for tentativa in range(1, max_tentativas + 1):
            try:
                ftp = ftplib.FTP("ftp.datasus.gov.br")
                ftp.login()
                ftp.cwd(diretorio_remoto)
                with open(caminho_dbc, "wb") as f:
                    ftp.retrbinary(f"RETR {arquivo}", f.write)
                ftp.quit()

                datasus_dbc.decompress(caminho_dbc, caminho_dbf)
                print(f"  ✓ {arquivo}")
                ok = True
                sucesso += 1
                break
            except Exception as e:
                print(f"  ✗ {arquivo} (tentativa {tentativa}/{max_tentativas}): {e}")
                # Remove o DBC pela metade antes de tentar de novo.
                if os.path.exists(caminho_dbc):
                    os.remove(caminho_dbc)
                time.sleep(espera_entre_tentativas)

        if not ok:
            falhas.append(arquivo)

    print(f"\n  Concluído: {sucesso}/{len(arquivos_alvo)} arquivos prontos.")
    if falhas:
        print(f"  ⚠ FALHARAM {len(falhas)} arquivos mesmo após "
              f"{max_tentativas} tentativas cada:")
        for f in falhas:
            print(f"    - {f}")
        print("  Rode este script NOVAMENTE — os prontos são pulados, só os "
              "que falharam são tentados de novo.")
    else:
        print("  Todos os arquivos baixados e convertidos com sucesso.")

    return falhas


def unir_arquivos_dbf(diretorio_local):
    """Junta todos os DBF locais num único DataFrame.

    Converte as colunas categóricas arquivo a arquivo (antes do concat) para manter
    o pico de memória baixo, e anota estado e ano a partir do nome do arquivo.
    """
    dataframes = []
    for arquivo in os.listdir(diretorio_local):
        if arquivo.endswith(".dbf"):
            caminho_arquivo = os.path.join(diretorio_local, arquivo)
            table = DBF(caminho_arquivo, encoding="latin-1")
            df = pd.DataFrame(iter(table))

            estado = arquivo[2:4]   # "DOAC2019.dbf" → "AC"
            ano    = arquivo[4:8]   # "DOAC2019.dbf" → "2019"
            df["ESTADO"] = estado
            df["ANO"]    = ano

            for col in COLUNAS_CATEGORICAS:
                if col in df.columns:
                    df[col] = df[col].astype("category")

            dataframes.append(df)
            print(f"Processado: {arquivo}")

    df_final = pd.concat(dataframes, ignore_index=True)
    return df_final


os.makedirs(DIR_DADOS, exist_ok=True)

# ── Download ───────────────────────────────────────────────────────────────────
falhas = baixar_e_converter_arquivos(DIRETORIO_REMOTO, DIRETORIO_LOCAL, ANOS, ESTADOS)

# ── Verificação ────────────────────────────────────────────────────────────────
arquivos_dbf = [f for f in os.listdir(DIRETORIO_LOCAL) if f.endswith(".dbf")]
esperado     = len(ANOS) * len(ESTADOS)
print(f"\n{'='*60}")
print(f"  VERIFICAÇÃO: {len(arquivos_dbf)}/{esperado} arquivos .dbf no diretório local")
print(f"{'='*60}")
if len(arquivos_dbf) < esperado:
    print("  ⚠ Download incompleto. Rode este script novamente antes de prosseguir.")
else:
    print("  ✓ Completo — pode prosseguir.")

# ── União de todos os anos/estados num só DataFrame ────────────────────────────
df_unificado = unir_arquivos_dbf(DIRETORIO_LOCAL)
print(df_unificado.head())

df_unificado.to_csv(
    DIR_DADOS + "mortalidade_unificada.csv",
    index=False, sep=";", encoding="utf-8", escapechar="\\", quoting=1,
)
print(f"Arquivo unificado salvo como '{DIR_DADOS}mortalidade_unificada.csv'.")

# ── Recorte de colunas → df_filtrado ───────────────────────────────────────────
colunas_existentes = [c for c in COLUNAS_SELECIONADAS if c in df_unificado.columns]
df_filtrado = df_unificado[colunas_existentes].copy()

for col in COLUNAS_CATEGORICAS:
    if col in df_filtrado.columns:
        df_filtrado[col] = df_filtrado[col].astype("category")

print(df_filtrado.info(memory_usage="deep"))
print(df_filtrado.head())

# Libera o DataFrame grande (todas as causas) — daqui pra frente só o filtrado.
del df_unificado
gc.collect()
print("\n  df_unificado liberado da memória (del + gc.collect()).")

# ── Persiste df_filtrado (todos os óbitos, colunas selecionadas) ───────────────
df_filtrado.to_csv(
    DIR_DADOS + "filtrado.csv",
    index=False, sep=";", encoding="utf-8", escapechar="\\", quoting=1,
)
df_filtrado.to_parquet(DIR_DADOS + "filtrado.parquet", index=False)
print(f"  df_filtrado salvo em {DIR_DADOS}filtrado.csv e {DIR_DADOS}filtrado.parquet.")
print(f"\n  df_filtrado pronto: {len(df_filtrado):,} óbitos "
      f"({len(ESTADOS)} estados × {len(ANOS)} anos).")

# ── Isola os homicídios confirmados (CID-10 X85–Y09) ───────────────────────────
df_homicidios = df_filtrado[
    df_filtrado["CAUSABAS_O"].astype(str).str[:3].isin(CID10_HOMICIDIO)
].copy().reset_index(drop=True)

print(f"\ndf_homicidios: {len(df_homicidios):,} registros "
      f"(CID-10 X85–Y09, homicídios confirmados)")
for est in ESTADOS:
    sub = df_homicidios[df_homicidios["ESTADO"] == est]
    print(f"  {est}: {len(sub):,} registros")

# Este parquet é a entrada de todas as etapas seguintes (02–08).
df_homicidios.to_parquet(DIR_DADOS + "homicidios.parquet", index=False)
print(f"\n  df_homicidios salvo em {DIR_DADOS}homicidios.parquet.")

# Dados novos → o cache de features OHE ficou velho; apaga para a 05 reconstruir.
if os.path.exists("features_cache.pkl"):
    os.remove("features_cache.pkl")
    print("  Cache de features (features_cache.pkl) invalidado — será reconstruído em 05.")
