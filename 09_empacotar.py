import glob
import os
import platform
import subprocess
import zipfile
from collections import Counter
from datetime import datetime

from config import DIR_FIGURAS, DIR_RESULTADOS, ESTADOS

# True = gera também o ZIP com TODAS as figuras (pesado); False = só o essencial.
GERAR_ZIP_COMPLETO = True

os.makedirs(DIR_RESULTADOS, exist_ok=True)

pngs    = glob.glob(DIR_FIGURAS + "*.png")
tem_log = os.path.exists(DIR_RESULTADOS + "log_resultados.txt")

# ══════════════════════════════════════════════════════════════════════════════
#  Conjunto ESSENCIAL — só as figuras que entram direto na escrita do TCC
# ══════════════════════════════════════════════════════════════════════════════
# Sempre entram: Hopkins, distribuições (linhas_) e perfis (perfil_).
essenciais = set()
for png in pngs:
    base = os.path.basename(png)
    if base.startswith(("hopkins", "linhas_", "perfil_")):
        essenciais.add(png)

# UMAP: um por algoritmo, do estado de maior volume disponível.
PRIORIDADE_ESTADO = ["PA", "MA", "AM", "MT", "RO", "TO", "AC", "AP", "RR"]
for algo in ["kmeans", "dbscan", "hdbscan", "aglomerativo"]:
    for est in PRIORIDADE_ESTADO:
        cand = f"{DIR_FIGURAS}{algo}_umap_{est}.png"
        if os.path.exists(cand):
            essenciais.add(cand)
            break


def _prefixo(nome):
    """Classifica uma figura pelo prefixo do nome (para contar por tipo no resumo)."""
    base = os.path.basename(nome)
    for p in ("hopkins", "kmeans", "dbscan", "hdbscan",
              "aglomerativo", "linhas", "perfil"):
        if base.startswith(p):
            return p
    return "outros"


def montar_zip(caminho_zip, lista_pngs, incluir_log):
    """Monta um ZIP direto dos arquivos (sem cópia intermediária); retorna o tamanho em MB.

    As imagens entram sem recompressão (ZIP_STORED, PNG já é comprimido); o log de
    texto entra comprimido (ZIP_DEFLATED).
    """
    with zipfile.ZipFile(caminho_zip, 'w') as zf:
        for png in lista_pngs:
            zf.write(png, arcname=os.path.basename(png),
                     compress_type=zipfile.ZIP_STORED)
        if incluir_log and tem_log:
            log_path = DIR_RESULTADOS + "log_resultados.txt"
            zf.write(log_path, arcname="log_resultados.txt",
                     compress_type=zipfile.ZIP_DEFLATED)
    return os.path.getsize(caminho_zip) / 1024 / 1024


timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# ── ZIP ESSENCIAL ─────────────────────────────────────────────────────────────
zip_ess  = DIR_RESULTADOS + f"essencial_{timestamp}.zip"
ess_list = sorted(essenciais)
print(f"Montando ZIP ESSENCIAL ({len(ess_list)} imagens + log)...")
mb_ess = montar_zip(zip_ess, ess_list, incluir_log=True)

print(f"\n{'='*70}")
print(f"  ZIP ESSENCIAL: {zip_ess}")
print(f"  Tamanho: {mb_ess:.1f} MB | {len(ess_list)} imagens + log_resultados.txt")
print(f"  Conteúdo (o que NÃO está no txt):")
for p, c in sorted(Counter(_prefixo(x) for x in ess_list).items()):
    print(f"    {p:16s}: {c}")
print(f"{'='*70}")

# ── ZIP COMPLETO (todas as figuras) — só se GERAR_ZIP_COMPLETO ────────────────
if GERAR_ZIP_COMPLETO:
    zip_full = DIR_RESULTADOS + f"completo_{timestamp}.zip"
    print(f"\nMontando ZIP COMPLETO ({len(pngs)} imagens + log)...")
    mb_full = montar_zip(zip_full, pngs, incluir_log=True)
    print(f"\n{'='*70}")
    print(f"  ZIP COMPLETO: {zip_full}")
    print(f"  Tamanho: {mb_full:.1f} MB | {len(pngs)} imagens + log")
    for p, c in sorted(Counter(_prefixo(x) for x in pngs).items()):
        print(f"    {p:16s}: {c}")
    print(f"{'='*70}")
else:
    print(f"\n  (ZIP completo NÃO gerado — {len(pngs)} PNGs disponíveis em {DIR_FIGURAS})")
    print(f"   Para gerar o pacote com todas as figuras do TCC, mude")
    print(f"   GERAR_ZIP_COMPLETO = True no topo deste script e rode de novo.")

# ── Abre a pasta de resultados no explorador de arquivos do SO ────────────────
pasta_abs = os.path.abspath(DIR_RESULTADOS)
print(f"\nResultados em: {pasta_abs}")
if platform.system() == "Windows":
    subprocess.Popen(f'explorer "{pasta_abs}"')
elif platform.system() == "Darwin":
    subprocess.Popen(["open", pasta_abs])
else:
    subprocess.Popen(["xdg-open", pasta_abs])
