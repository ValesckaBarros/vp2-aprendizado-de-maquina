"""Gera o PDF do relatório VP2 (capa, sumário, lista de figuras) a partir do Markdown.

Uso: python3 build.py <RELATORIO.md> <pasta_figuras> <saida.pdf>
Duas passagens no Chromium: a 1ª descobre em que página cai cada seção/figura,
a 2ª grava o sumário com os números de página corretos.
"""
import os, re, subprocess, sys, shutil
from bs4 import BeautifulSoup

MD, FIGS, OUT = sys.argv[1:4]
BUILD = os.path.dirname(os.path.abspath(__file__))
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

CAPTIONS = {
    "cast_dist": "Casting: distribuição das classes (treino + teste)",
    "cast_amostras": "Casting: amostras originais de cada classe",
    "cast_media_hist": "Casting: imagem média por classe e histograma de intensidade dos pixels",
    "cast_hog_exemplo": "Casting: imagem 64×64 e sua representação HOG",
    "cast_pca": "Casting: variância explicada acumulada do PCA sobre o HOG",
    "cast_gs_vs_optuna": "Casting: melhor F1 e tempo de busca, GridSearch × Optuna",
    "cast_confusao": "Casting: matrizes de confusão no teste (715 imagens)",
    "cast_cnn_curvas": "Casting: curvas de loss e acurácia da CNN final",
    "cast_cnn_erros": "Casting: imagens classificadas erradamente pela CNN",
    "neu_amostras": "NEU: amostras originais das seis classes",
    "neu_media_hist": "NEU: imagem média por classe e histograma de intensidade dos pixels",
    "neu_pca": "NEU: variância explicada acumulada do PCA sobre o HOG",
    "neu_gs_vs_optuna": "NEU: melhor F1 e tempo de busca, GridSearch × Optuna",
    "neu_confusao": "NEU: matrizes de confusão no teste (270 imagens)",
    "neu_cnn_curvas": "NEU: curvas de loss e acurácia da CNN final",
    "neu_f1_por_classe": "NEU: F1 por classe e por modelo",
    "neu_cnn_erros": "NEU: imagens classificadas erradamente pela CNN",
    "comparacao_cruzada": "Comparação cruzada: F1 macro no teste e tempo de treino",
    "rnn_ciclos_por_motor": "C-MAPSS: distribuição do nº de ciclos por motor (treino e teste)",
    "rnn_series_sensores": "C-MAPSS: séries temporais de três sensores em três motores",
    "rnn_correlacao": "C-MAPSS: matriz de correlação dos 14 sensores úteis e do RUL",
    "rnn_rul_clip": "C-MAPSS: alvo RUL linear e piece-wise (clip em 130)",
    "rnn_loss_baseline": "RNN: curvas de loss da LSTM de partida (2×64, W = 30)",
    "rnn_optuna_importancia": "Optuna: importância dos hiperparâmetros (fANOVA)",
    "rnn_optuna_historico": "Optuna: histórico da otimização",
    "rnn_optuna_W_celula": "Optuna: val RMSE por tamanho de janela W e por tipo de célula",
    "rnn_loss_best": "RNN: curvas de loss do melhor modelo (GRU 2×96, W = 50)",
    "rnn_real_vs_previsto": "RNN: RUL real × previsto nos 100 motores de teste",
    "rnn_scatter": "RNN: dispersão RUL real × previsto",
    "rnn_analise_erros": "RNN: análise dos erros por faixa de RUL e tamanho da série",
}

# ---------- Markdown -> HTML ----------
md = open(MD, encoding="utf-8").read()
md = md[md.index("## 1. Introdução"):]           # título e autoria vão para a capa
md = md.replace("](figuras/", f"]({FIGS}/")
body_html = subprocess.run(["pandoc", "-f", "markdown", "-t", "html", "--no-highlight", "--wrap=none"],
                           input=md, capture_output=True, text=True, check=True).stdout
soup = BeautifulSoup(body_html, "html.parser")

figs = []
for i, fig in enumerate(soup.find_all("figure"), 1):
    key = os.path.splitext(os.path.basename(fig.img["src"]))[0]
    cap = CAPTIONS[key]
    fig["id"] = f"fig{i}"
    fig.figcaption.clear()
    fig.figcaption.append(BeautifulSoup(f"<b>Figura {i}</b> – {cap}", "html.parser"))
    figs.append((f"fig{i}", f"Figura {i} – {cap}", f"Figura {i} – {cap}"))

heads = []
for h in soup.find_all(["h2", "h3"]):
    heads.append((h.name, h.get("id"), h.get_text(" ", strip=True)))
for t in soup.find_all("table"):
    t.wrap(soup.new_tag("div", attrs={"class": "tbl"}))

COVER = """
<section class="cover">
  <div class="inst">UNIVERSIDADE FEDERAL DE PERNAMBUCO<br>
  CENTRO DE TECNOLOGIA E GEOCIÊNCIAS<br>
  PROGRAMA DE PÓS-GRADUAÇÃO EM ENGENHARIA DE PRODUÇÃO (PPGEP)</div>
  <div class="author">VALESCKA BARROS</div>
  <div class="title">Deep Learning na inspeção industrial e na manutenção preditiva</div>
  <div class="subtitle">CNN × modelos clássicos na detecção de defeitos e RNN para estimar a vida útil remanescente de motores</div>
  <div class="nature">Relatório técnico da Segunda Verificação Parcial da disciplina
  Aprendizado de Máquina, ministrada pelo Prof. Márcio das Chagas Moura.</div>
  <div class="place">Recife<br>2026</div>
</section>
"""

def lists_html(pages):
    def pg(key):
        return pages.get(key, "00")
    toc = ['<section class="toc"><h1 class="front">Sumário</h1>']
    for lvl, hid, text in heads:
        toc.append(f'<div class="toc-{lvl}"><a href="#{hid}"><span class="t">{text}</span>'
                   f'<span class="dots"></span><span class="p">{pg(("h", hid))}</span></a></div>')
    toc.append('</section><section class="toc"><h1 class="front">Lista de figuras</h1>')
    for fid, text, _ in figs:
        toc.append(f'<div class="toc-fig"><a href="#{fid}"><span class="t">{text}</span>'
                   f'<span class="dots"></span><span class="p">{pg(("f", fid))}</span></a></div>')
    toc.append("</section>")
    return "\n".join(toc)

CSS = open(os.path.join(BUILD, "style.css"), encoding="utf-8").read()

def render(pages, pdf):
    html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório VP2 – Aprendizado de Máquina</title><style>{CSS}</style></head>
<body>{COVER}{lists_html(pages)}<main>{soup}</main></body></html>"""
    path = os.path.join(BUILD, "relatorio.html")
    open(path, "w", encoding="utf-8").write(html)
    subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", "file://" + path],
                   check=True, capture_output=True)

def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

# ---------- passagem 1: descobre as páginas ----------
tmp = os.path.join(BUILD, "pass1.pdf")
render({}, tmp)
txt = subprocess.run(["pdftotext", "-layout", tmp, "-"], capture_output=True, text=True, check=True).stdout
page_txt = [norm(p) for p in txt.split("\f")]
first_body = next(i for i, p in enumerate(page_txt) if "1. introdução" in p and "sumário" not in p
                  and "lista de figuras" not in p and "este trabalho" in p)

pages = {}
def locate(key, text):
    for n in (45, 22):
        probe = norm(text)[:n]
        for i in range(first_body, len(page_txt)):
            if probe in page_txt[i]:
                pages[key] = str(i + 1)
                return
    print("NÃO ENCONTRADO:", text)

for lvl, hid, text in heads:
    locate(("h", hid), text)
for fid, _, text in figs:
    locate(("f", fid), text)

# ---------- passagem 2: sumário com páginas ----------
render(pages, OUT)
n = subprocess.run(["pdfinfo", OUT], capture_output=True, text=True).stdout
print(re.search(r"Pages:\s+\d+", n).group(0), "| itens localizados:", len(pages), "de", len(heads) + len(figs))
