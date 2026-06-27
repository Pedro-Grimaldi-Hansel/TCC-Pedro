import os
import pickle

import pandas as pd

from funcoes import preparar_features_ohe
from config import ESTADOS

# Cache do OHE em disco. Fica na raiz porque é compartilhado entre processos:
# cada etapa (05–08) roda isolada via runpy, então não dá para passar o objeto
# em memória — ele é gravado aqui e relido pela etapa seguinte.
_CACHE_PATH = "features_cache.pkl"


def build_cache(df_homicidios: pd.DataFrame) -> dict:
    """Devolve o OHE por estado, reaproveitando o cache em disco se existir.

    Retorna {estado: (X_scaled, features, df_mod)}. Como `preparar_features_ohe`
    é determinístico, o resultado é o mesmo entre as etapas 05–08 — daí valer a
    pena calcular uma vez (na 05) e reler nas demais, em vez de refazer o OHE.
    A etapa 01 apaga este arquivo sempre que há dados novos.
    """
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"  Cache de features carregado: {_CACHE_PATH} "
              f"({len(cache)} estados)")
        return cache

    # Sem cache: constrói o OHE de cada estado uma vez e persiste.
    cache = {}
    for estado in ESTADOS:
        df_e = df_homicidios[df_homicidios["ESTADO"] == estado].copy()
        if df_e.empty:
            continue
        X_scaled, features, df_mod = preparar_features_ohe(df_e)
        cache[estado] = (X_scaled, features, df_mod)
        print(f"  Cache: {estado} ({len(X_scaled):,} registros, "
              f"{len(features)} features)")

    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Cache de features salvo: {_CACHE_PATH}")
    return cache
