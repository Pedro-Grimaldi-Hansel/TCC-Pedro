import sys


class _Tee:
    """Espelha o que é impresso em vários destinos ao mesmo tempo.

    Usado para que cada `print` apareça no console E vá para o arquivo de log
    numa só passada, sem precisar duplicar chamadas pelo pipeline.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, texto):
        for s in self.streams:
            try:
                s.write(texto)
            except UnicodeEncodeError:
                # Se o console estiver num encoding limitado (cp1252 no Windows),
                # substitui os caracteres que ele não representa em vez de
                # interromper a execução do pipeline inteiro.
                enc = getattr(s, "encoding", None) or "ascii"
                s.write(texto.encode(enc, errors="replace").decode(enc))
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def iniciar_log(caminho="log_resultados.txt"):
    """Redireciona o stdout para console + arquivo, em UTF-8.

    Chamada uma única vez no `main.py`. Força UTF-8 para que acentos e o
    caractere █ (usado nos banners) saiam corretos mesmo no console do Windows;
    o `_Tee` ainda tem um fallback caso essa reconfiguração não seja possível.
    Abre o log em modo "w" — cada execução começa um log limpo.
    """
    for _s in (sys.__stdout__, sys.__stderr__):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    log_file = open(caminho, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print("=" * 70)
    print(f"LOG INICIADO — {caminho}")
    print("=" * 70)
