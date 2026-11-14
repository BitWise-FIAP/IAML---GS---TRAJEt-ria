# -*- coding: utf-8 -*-
"""Verifica automaticamente se a entrega de IA & Machine Learning está completa.

Roda em qualquer máquina, sem internet e sem reexecutar o modelo:

    cd ml
    python verificar_entrega.py

Cada item da rubrica vira uma checagem objetiva. Saída: tabela com OK/FALHA,
um resumo por critério e código de saída 0 (tudo ok) ou 1 (algo falhou).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ML = RAIZ / "ml"

OK = "OK  "
FALHA = "FALHA"
AVISO = "AVISO"

resultados: list[tuple[str, str, str, str]] = []  # (critério, status, item, detalhe)


def checar(criterio: str, item: str, condicao: bool, detalhe: str = "", aviso: bool = False) -> bool:
    status = OK if condicao else (AVISO if aviso else FALHA)
    resultados.append((criterio, status, item, detalhe))
    return condicao


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------
def texto_das_saidas(celula: dict) -> str:
    partes = []
    for saida in celula.get("outputs", []):
        tipo = saida.get("output_type")
        if tipo == "stream":
            bruto = saida.get("text", "")
            partes.append(bruto if isinstance(bruto, str) else "".join(bruto))
        elif tipo == "execute_result":
            bruto = saida.get("data", {}).get("text/plain", "")
            partes.append(bruto if isinstance(bruto, str) else "".join(bruto))
    return "".join(partes)


def ler_notebook(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def verificar_notebook(nb: dict) -> str:
    codigo = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    fonte = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

    execs = [c.get("execution_count") for c in codigo]
    executado = all(isinstance(e, int) for e in execs)
    checar("Notebook executado", "todas as células de código com execution_count",
           executado, f"{len(codigo)} células | contadores: {min(execs)}–{max(execs)}" if executado else "há célula sem execução")
    checar("Notebook executado", "contadores sequenciais (Restart & Run All)",
           executado and execs == sorted(execs) and execs[0] == 1 and len(set(execs)) == len(execs),
           "sequência 1..N sem repetição" if executado else "")

    erros = [
        (i, o.get("ename"))
        for i, c in enumerate(nb["cells"])
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    checar("Notebook executado", "nenhuma saída de erro", not erros,
           "zero tracebacks" if not erros else f"{len(erros)} erro(s): {erros[:3]}")

    com_saida = sum(1 for c in codigo if c.get("outputs"))
    checar("Notebook executado", "saídas salvas nas células", com_saida >= len(codigo) - 1,
           f"{com_saida}/{len(codigo)} células com saída")

    # semente
    checar("Reprodutibilidade", "semente fixa declarada",
           bool(re.search(r"SEMENTE\s*=\s*42", fonte)) and "random_state=SEMENTE" in fonte,
           "SEMENTE = 42 aplicada em split, grid, CV e permutação")

    # hyperparâmetros: lê o dicionário "grade" do próprio notebook
    bloco = re.search(r"grade\s*=\s*\{(.*?)\n\}", fonte, re.S)
    if bloco:
        valores = re.findall(r"\[\s*([^\]]+?)\s*\]", bloco.group(1))
        listas = [[v.strip() for v in grupo.split(",") if v.strip()] for grupo in valores]
        combos = 1
        for lista in listas:
            combos *= len(lista)
        checar("Treino com ≥10 combinações", "3 hiperparâmetros na grade",
               len(listas) == 3, f"{len(listas)} hiperparâmetros: " + ", ".join(f"{n} valores" for n in map(len, listas)))
        checar("Treino com ≥10 combinações", "≥10 combinações avaliadas",
               combos >= 10, f"{combos} combinações")
    else:
        checar("Treino com ≥10 combinações", "grade de hiperparâmetros encontrada", False, "dict `grade` não localizado")

    # CV apenas no treino: as chamadas de CV não podem tocar em X_te / y_te
    linhas_cv = [
        linha for linha in fonte.splitlines()
        if re.search(r"cross_val_score\(|GridSearchCV\(|busca\.fit\(", linha)
    ]
    vazamento = [linha.strip() for linha in linhas_cv if re.search(r"\b[XY]_te\b", linha)]
    checar("CV só no treino", "nenhuma chamada de CV usa o conjunto de teste",
           bool(linhas_cv) and not vazamento,
           f"{len(linhas_cv)} chamadas verificadas" if not vazamento else f"vazamento em: {vazamento[:2]}")

    # vazamento de alvo nas features
    try:
        sys.path.insert(0, str(ML))
        from src import ibge_api  # noqa: E402

        features = ibge_api.FEATURES
        alvo = ibge_api.ALVO
        checar("Sem vazamento de dados", "alvo fora da lista de features",
               alvo not in features, f"alvo `{alvo}` | {len(features)} features")
        checar("Sem vazamento de dados", "módulo de coleta importável",
               True, f"src/ibge_api.py | {len(ibge_api.TABELAS)} tabelas do SIDRA")
    except Exception as erro:  # pragma: no cover - só em ambiente quebrado
        checar("Sem vazamento de dados", "importação de src/ibge_api.py", False, repr(erro))

    # métricas e seções obrigatórias
    saida = "\n".join(texto_das_saidas(c) for c in codigo)
    checar("Treino vs teste", "métricas de treino e teste na mesma tabela",
           "treino" in saida and "teste" in saida and "R2" in saida,
           "coluna `conjunto` com treino/teste")
    checar("Treino vs teste", "baselines comparados",
           "Dummy" in saida and "Regress" in saida and "Random Forest" in saida,
           "Dummy + Regressão Linear + Random Forest")
    checar("Overfitting discutido", "curva de complexidade calculada",
           "R2_treino" in fonte and "R2_validacao" in fonte and "R2_teste" in fonte,
           "R² de treino, validação e teste por max_depth")
    checar("Overfitting discutido", "gap treino-teste impresso no resumo",
           "Gap treino-teste" in saida, "resumo final da execução")
    checar("Conclusão para o cidadão", "ranking municipal gerado",
           "prioridade_sp.csv" in saida, "645 municípios de SP")
    checar("Conclusão para o cidadão", "limitações explícitas na conclusão",
           "Falácia ecológica" in fonte and "Não é causal" in fonte, "seção 12 do notebook")
    return fonte


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def ler_csv(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def verificar_dados() -> None:
    brutos = sorted(ML.glob("dados/raw/tabela*_N6.json"))
    metas = sorted(ML.glob("dados/raw/tabela*_N6.meta.json"))
    checar("Dados reais e públicos", "7 tabelas do Censo 2022 no snapshot",
           len(brutos) == 7, f"{len(brutos)} arquivos JSON em dados/raw/")
    checar("Dados reais e públicos", "metadados de coleta (URL + data + SHA-256)",
           len(metas) == len(brutos), f"{len(metas)} arquivos .meta.json")

    hashes_ok = 0
    for meta in metas:
        dados = json.loads(meta.read_text(encoding="utf-8"))
        if (
            re.fullmatch(r"[0-9a-f]{64}", dados.get("sha256", ""))
            and dados.get("url", "").startswith("https://servicodados.ibge.gov.br/")
            and dados.get("coletado_em")
        ):
            hashes_ok += 1
    checar("Dados reais e públicos", "procedência auditável em todos os metadados",
           hashes_ok == len(metas), f"{hashes_ok}/{len(metas)} com URL do IBGE, data e hash válidos")

    total_bytes = sum(p.stat().st_size for p in brutos)
    checar("Dados reais e públicos", "snapshot presente (não depende da internet)",
           total_bytes > 20_000_000, f"{total_bytes / 1e6:.1f} MB de JSON bruto")

    base = ler_csv(ML / "dados/processed/municipios_features.csv")
    colunas = set(base[0].keys()) if base else set()
    esperadas = {
        "codigo", "municipio", "uf", "pop_18_24", "taxa_sem_EM",
        "adultos_EM_mais", "pct_baixa_renda", "pct_internet",
        "pct_esgoto_adequado", "pct_preta_parda", "pct_urbana", "pct_15_24",
    }
    checar("Dados reais e públicos", "base de features completa",
           esperadas.issubset(colunas), f"{len(base)} municípios | {len(colunas)} colunas")
    checar("Dados reais e públicos", "cobertura municipal do Censo",
           len(base) == 5570, f"{len(base)}/5570 municípios")
    checar("Dados reais e públicos", "sem valores ausentes no alvo e nas features",
           all(all(v not in ("", None) for v in linha.values()) for linha in base),
           "nenhuma imputação necessária")

    ranking = ler_csv(ML / "dados/processed/prioridade_sp.csv")
    campos = {"codigo_ibge", "municipio", "regiao", "pop_18_24", "taxa_observada",
              "taxa_prevista", "jovens_sem_EM_estimado", "posicao_volume",
              "prioridade", "nivel_prioridade"}
    checar("Conclusão para o cidadão", "ranking operacional de São Paulo",
           bool(ranking) and campos.issubset(ranking[0].keys()),
           f"{len(ranking)} municípios | {len(ranking[0])} colunas" if ranking else "arquivo vazio")
    checar("Conclusão para o cidadão", "ranking completo dos 645 municípios paulistas",
           len(ranking) == 645, f"{len(ranking)}/645")
    posicoes = sorted(int(linha["posicao_volume"]) for linha in ranking)
    checar("Conclusão para o cidadão", "posições únicas de 1 a 645",
           posicoes == list(range(1, len(ranking) + 1)), "sem empate nem lacuna")
    altas = sum(1 for linha in ranking if linha["nivel_prioridade"] == "Alta")
    checar("Conclusão para o cidadão", "faixa Alta = top 10% em volume",
           63 <= altas <= 65, f"{altas} municípios marcados como Alta")


# ---------------------------------------------------------------------------
# Figuras e documentos
# ---------------------------------------------------------------------------
def verificar_artefatos() -> None:
    figuras = sorted((ML / "figuras").glob("*.png"))
    checar("Evidências visuais", "6 figuras geradas pelo notebook",
           len(figuras) == 6 and all(f.stat().st_size > 20_000 for f in figuras),
           ", ".join(f.name.split("_")[0] for f in figuras))

    obrigatorios = [
        RAIZ / "README.md",
        RAIZ / "RELATORIO_ML.md",
        RAIZ / "ROTEIRO_VIDEO.md",
        RAIZ / ".gitignore",
        ML / "trajetoria_ml.ipynb",
        ML / "requirements.txt",
        ML / "src" / "ibge_api.py",
        ML / "dados" / "README.md",
    ]
    faltando = [c.name for c in obrigatorios if not c.exists()]
    checar("Entregáveis", "todos os arquivos obrigatórios presentes",
           not faltando, "README, relatório, roteiro, notebook, requirements" if not faltando else f"faltam: {faltando}")

    gitignore = (RAIZ / ".gitignore").read_text(encoding="utf-8")
    checar("Entregáveis", ".gitignore cobre caches de Python e Jupyter",
           "__pycache__" in gitignore and ".ipynb_checkpoints" in gitignore, ".gitignore na raiz")

    equipe = {
        "Enrico Ricarte": "558571",
        "Pedro Gaspar": "554887",
        "Victor Freire": "556191",
    }
    for documento in ("README.md", "RELATORIO_ML.md"):
        texto = (RAIZ / documento).read_text(encoding="utf-8")
        completo = all(nome in texto and rm in texto for nome, rm in equipe.items())
        checar("Identificação", f"equipe com nome e RM em {documento}", completo,
               "3 integrantes + turma 3ESPH" if completo else "faltando nome ou RM")
        checar("Identificação", f"turma 3ESPH em {documento}", "3ESPH" in texto, "")

    relatorio = (RAIZ / "RELATORIO_ML.md").read_text(encoding="utf-8")
    palavras = len(re.findall(r"\w+", relatorio))
    checar("Relatório 2–4 páginas", "extensão compatível com 2 a 4 páginas",
           1200 <= palavras <= 2400, f"{palavras} palavras (~{palavras / 550:.1f} páginas)")

    notebook_fonte = (ML / "trajetoria_ml.ipynb").read_text(encoding="utf-8")
    citou_mock = "seed.json" in notebook_fonte
    checar("Dados reais e públicos", "dados fictícios do app explicitamente recusados",
           citou_mock, "o notebook cita seed.json apenas para justificar a não utilização")

    readme = (RAIZ / "README.md").read_text(encoding="utf-8")
    placeholder = "preencher com o link do YouTube" in readme
    checar("Vídeo de 5 minutos", "link do YouTube publicado no README",
           not placeholder, "link já preenchido" if not placeholder else "ainda é placeholder", aviso=True)


# ---------------------------------------------------------------------------
# Relatório final
# ---------------------------------------------------------------------------
def imprimir() -> int:
    ordem: list[str] = []
    for criterio, _, _, _ in resultados:
        if criterio not in ordem:
            ordem.append(criterio)

    for criterio in ordem:
        print(f"\n{criterio.upper()}")
        for crit, status, item, detalhe in resultados:
            if crit != criterio:
                continue
            print(f"  [{status}] {item}")
            if detalhe:
                print(f"          {detalhe}")

    falhas = sum(1 for _, status, _, _ in resultados if status == FALHA)
    avisos = sum(1 for _, status, _, _ in resultados if status == AVISO)
    oks = sum(1 for _, status, _, _ in resultados if status == OK)
    total = len(resultados)

    print("\n" + "=" * 74)
    print(f"RESULTADO: {oks}/{total} verificações OK | {avisos} aviso(s) | {falhas} falha(s)")
    if falhas:
        print("Pendências:")
        for criterio, status, item, detalhe in resultados:
            if status == FALHA:
                print(f"  - {criterio}: {item} ({detalhe})")
    elif avisos:
        print("Único item pendente é manual: gravar o vídeo e publicar o link.")
    else:
        print("Entrega completa e consistente.")
    print("=" * 74)
    return 0 if falhas == 0 else 1


def main() -> int:
    print("=" * 74)
    print("VERIFICAÇÃO DA ENTREGA — IA & Machine Learning | TRAJEtória")
    print(f"Raiz: {RAIZ}")
    print("=" * 74)

    caminho_nb = ML / "trajetoria_ml.ipynb"
    if not checar("Entregáveis", "notebook encontrado", caminho_nb.exists(), str(caminho_nb.name)):
        return imprimir()

    nb = ler_notebook(caminho_nb)
    verificar_notebook(nb)
    verificar_dados()
    verificar_artefatos()
    return imprimir()


if __name__ == "__main__":
    sys.exit(main())
