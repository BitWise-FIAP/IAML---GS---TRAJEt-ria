"""Coleta e tratamento dos dados do Censo 2022 (IBGE / SIDRA) usados na TRAJEtória.

Este módulo concentra tudo o que o notebook precisa para sair do número bruto do
IBGE e chegar na tabela de features por município:

1. baixa cada tabela do agregado pela API v3 do SIDRA e guarda um snapshot em
   ``dados/raw/`` (com URL, data da coleta e SHA-256);
2. achata a resposta JSON em um DataFrame tidy
   (``codigo``, ``municipio``, ``uf``, ``valor`` e uma coluna ``c<classificação>``
   com o código da categoria);
3. combina as tabelas em uma linha por município, com o alvo e as features.

A partir da segunda execução o notebook lê o snapshot local e funciona offline.

Fontes (todas públicas, Censo Demográfico 2022, nível municipal N6):
    https://sidra.ibge.gov.br/tabela/10061
    https://sidra.ibge.gov.br/tabela/10296
    https://sidra.ibge.gov.br/tabela/9936
    https://sidra.ibge.gov.br/tabela/6805
    https://sidra.ibge.gov.br/tabela/9605
    https://sidra.ibge.gov.br/tabela/9923
    https://sidra.ibge.gov.br/tabela/9514
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"
PERIODO = "2022"
LOCALIDADES = "N6[all]"  # todos os municípios do Brasil

DIR_ML = Path(__file__).resolve().parents[1]
DIR_RAW = DIR_ML / "dados" / "raw"
DIR_PROCESSED = DIR_ML / "dados" / "processed"

# ---------------------------------------------------------------------------
# Catálogo das tabelas usadas. A chave é o número da tabela no SIDRA.
# As classificações foram restringidas para reduzir o tamanho da resposta:
# só as categorias que viram feature ou denominador.
# ---------------------------------------------------------------------------
TABELAS: dict[int, dict[str, Any]] = {
    10061: {
        "variavel": 2667,
        "classificacao": "58[100052,108866]|1568[all]|2[6794]|86[95251]",
        "descricao": "Pessoas de 18 anos ou mais, por nível de instrução e grupo de idade",
        "uso": "alvo (jovens 18-24 sem ensino médio completo) e escolaridade dos adultos",
    },
    10296: {
        "variavel": 13604,
        "classificacao": "2[6794]|86[95251]|386[all]",
        "descricao": "Pessoas de 10 anos ou mais, por classes de rendimento nominal mensal domiciliar per capita",
        "uso": "percentual de pessoas em domicílios de baixa renda",
    },
    9936: {
        "variavel": 381,
        "classificacao": "2072[all]|63[95826]|125[2932]",
        "descricao": "Domicílios particulares permanentes ocupados, por existência de conexão à Internet",
        "uso": "percentual de domicílios com internet",
    },
    6805: {
        "variavel": 381,
        "classificacao": "11558[all]",
        "descricao": "Domicílios particulares permanentes ocupados, por tipo de esgotamento sanitário",
        "uso": "percentual de domicílios com esgotamento adequado",
    },
    9605: {
        "variavel": 93,
        "classificacao": "86[all]",
        "descricao": "População residente, por cor ou raça",
        "uso": "percentual de população preta ou parda",
    },
    9923: {
        "variavel": 93,
        "classificacao": "1[all]",
        "descricao": "População residente, por situação do domicílio",
        "uso": "percentual de população urbana",
    },
    9514: {
        "variavel": 93,
        "classificacao": "2[6794]|287[93086,93087,100362]|286[113635]",
        "descricao": "População residente, por sexo e idade",
        "uso": "percentual de jovens de 15 a 24 anos na população",
    },
}


# ---------------------------------------------------------------------------
# 1. Coleta
# ---------------------------------------------------------------------------
def montar_url(tabela: int, variavel: int, classificacao: str) -> str:
    """Monta a URL da API v3 do SIDRA para um agregado municipal."""
    return (
        f"{BASE_URL}/{tabela}/periodos/{PERIODO}/variaveis/{variavel}"
        f"?localidades={LOCALIDADES}&classificacao={classificacao}"
    )


def _caminho_raw(tabela: int) -> Path:
    variavel = TABELAS[tabela]["variavel"]
    return DIR_RAW / f"tabela{tabela}_v{variavel}_{PERIODO}_N6.json"


def baixar_tabela(tabela: int, forcar: bool = False) -> dict[str, Any]:
    """Baixa (ou reaproveita do cache) uma tabela e devolve seus metadados.

    O cache é o snapshot que dá reprodutibilidade à entrega: o notebook pode ser
    reexecutado offline e o JSON guardado é exatamente o que o IBGE respondeu.
    """
    info = TABELAS[tabela]
    destino = _caminho_raw(tabela)
    destino.parent.mkdir(parents=True, exist_ok=True)

    if destino.exists() and not forcar:
        bruto = destino.read_bytes()
        return {
            "tabela": tabela,
            "descricao": info["descricao"],
            "arquivo": destino.name,
            "origem": "cache local",
            "bytes": len(bruto),
            "sha256": hashlib.sha256(bruto).hexdigest(),
        }

    url = montar_url(tabela, info["variavel"], info["classificacao"])
    resposta = requests.get(
        url,
        timeout=600,
        headers={"User-Agent": "TRAJEtoria-FIAP-GS2026/1.0"},
    )
    resposta.raise_for_status()
    dados = resposta.json()

    destino.write_text(
        json.dumps(dados, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "tabela": tabela,
        "descricao": info["descricao"],
        "url": url,
        "periodo": PERIODO,
        "nivel": LOCALIDADES,
        "coletado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arquivo": destino.name,
        "bytes": destino.stat().st_size,
        "sha256": hashlib.sha256(destino.read_bytes()).hexdigest(),
    }
    destino.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**meta, "origem": "API do IBGE"}


def baixar_todas(forcar: bool = False) -> pd.DataFrame:
    """Baixa todas as tabelas do catálogo e devolve um resumo da coleta."""
    return pd.DataFrame([baixar_tabela(t, forcar=forcar) for t in TABELAS])


# ---------------------------------------------------------------------------
# 2. Leitura e achatamento do JSON
# ---------------------------------------------------------------------------
def _valor_numerico(serie: pd.Series) -> pd.Series:
    """Converte os valores do SIDRA em número.

    O IBGE usa ``...`` (não disponível), ``-`` (zero absoluto) e ``X`` (sigilo).
    ``-`` vira 0; o resto vira NaN, para não inventar dado.
    """
    limpo = serie.astype("string").str.strip()
    limpo = limpo.replace({"-": "0", "...": None, "X": None, "..": None})
    return pd.to_numeric(limpo, errors="coerce")


def ler_tabela(tabela: int) -> pd.DataFrame:
    """Lê o snapshot de uma tabela e devolve um DataFrame tidy.

    Colunas: ``variavel``, ``codigo``, ``municipio``, ``uf``, ``valor`` e uma
    coluna ``c<id>`` por classificação, contendo o código da categoria.
    """
    caminho = _caminho_raw(tabela)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Snapshot ausente: {caminho}. Rode `baixar_todas()` primeiro."
        )
    payload = json.loads(caminho.read_text(encoding="utf-8"))

    registros: list[dict[str, Any]] = []
    for variavel in payload:
        for resultado in variavel.get("resultados", []):
            categorias = {}
            for classificacao in resultado.get("classificacoes", []):
                codigo_categoria = next(iter(classificacao["categoria"]))
                categorias[f"c{classificacao['id']}"] = str(codigo_categoria)
            for serie in resultado.get("series", []):
                localidade = serie["localidade"]
                nome = localidade["nome"]
                uf = nome.rsplit(" - ", 1)[-1] if " - " in nome else ""
                registros.append(
                    {
                        "variavel": str(variavel["id"]),
                        "codigo": str(localidade["id"]),
                        "municipio": nome,
                        "uf": uf,
                        **categorias,
                        "valor": next(iter(serie["serie"].values())),
                    }
                )

    df = pd.DataFrame(registros)
    df["valor"] = _valor_numerico(df["valor"])
    return df


# ---------------------------------------------------------------------------
# 3. Engenharia de features
# ---------------------------------------------------------------------------
IDX = ["codigo", "municipio"]


def _proporcao(numerador: pd.Series, denominador: pd.Series, nome: str) -> pd.Series:
    return (numerador / denominador).rename(nome)


def construir_features() -> pd.DataFrame:
    """Combina as sete tabelas em uma linha por município.

    Retorna as colunas de identificação, o alvo ``taxa_sem_EM`` (proporção de
    jovens de 18 a 24 anos sem ensino médio completo), a população de 18 a 24
    anos (peso e volume) e as sete features explicativas.
    """
    # --- 10061: alvo, escolaridade adulta e população jovem -----------------
    t = ler_tabela(10061)
    t = t[(t["c2"] == "6794") & (t["c86"] == "95251")]

    jovens = t[t["c58"] == "100052"].pivot_table(
        index=IDX, columns="c1568", values="valor", aggfunc="sum"
    )
    pop_18_24 = jovens["120704"].rename("pop_18_24")
    sem_ensino_medio = jovens["9493"].fillna(0) + jovens["9494"].fillna(0)
    taxa_sem_em = _proporcao(sem_ensino_medio, jovens["120704"], "taxa_sem_EM")

    adultos = t[t["c58"] == "108866"].pivot_table(
        index=IDX, columns="c1568", values="valor", aggfunc="sum"
    )
    adultos_em_mais = _proporcao(
        adultos["9495"].fillna(0) + adultos["99713"].fillna(0),
        adultos["120704"],
        "adultos_EM_mais",
    )

    # --- 10296: renda domiciliar per capita --------------------------------
    r = ler_tabela(10296)
    r = r[(r["c2"] == "6794") & (r["c86"] == "95251")]
    renda = r.pivot_table(index=IDX, columns="c386", values="valor", aggfunc="sum")
    pct_baixa_renda = _proporcao(
        renda["9692"].fillna(0) + renda["9681"].fillna(0) + renda["9682"].fillna(0),
        renda["9680"],
        "pct_baixa_renda",
    )

    # --- 9936: conexão domiciliar à internet -------------------------------
    i = ler_tabela(9936)
    i = i[(i["c63"] == "95826") & (i["c125"] == "2932")]
    internet = i.pivot_table(index=IDX, columns="c2072", values="valor", aggfunc="sum")
    pct_internet = _proporcao(
        internet["77585"], internet["77585"] + internet["77586"], "pct_internet"
    )

    # --- 6805: esgotamento sanitário ---------------------------------------
    e = ler_tabela(6805)
    esgoto = e.pivot_table(index=IDX, columns="c11558", values="valor", aggfunc="sum")
    pct_esgoto_adequado = _proporcao(
        esgoto["46290"].fillna(0) + esgoto["72112"].fillna(0),
        esgoto["46292"],
        "pct_esgoto_adequado",
    )

    # --- 9605: cor ou raça --------------------------------------------------
    c = ler_tabela(9605)
    raca = c.pivot_table(index=IDX, columns="c86", values="valor", aggfunc="sum")
    pct_preta_parda = _proporcao(
        raca["2777"].fillna(0) + raca["2779"].fillna(0),
        raca["95251"],
        "pct_preta_parda",
    )

    # --- 9923: situação do domicílio ---------------------------------------
    s = ler_tabela(9923)
    situacao = s.pivot_table(index=IDX, columns="c1", values="valor", aggfunc="sum")
    pct_urbana = _proporcao(situacao["1"], situacao["6795"], "pct_urbana")

    # --- 9514: estrutura etária --------------------------------------------
    a = ler_tabela(9514)
    a = a[a["c2"] == "6794"]
    idade = a.pivot_table(index=IDX, columns="c287", values="valor", aggfunc="sum")
    pct_15_24 = _proporcao(
        idade["93086"].fillna(0) + idade["93087"].fillna(0),
        idade["100362"],
        "pct_15_24",
    )

    df = pd.concat(
        [
            pop_18_24,
            taxa_sem_em,
            adultos_em_mais,
            pct_baixa_renda,
            pct_internet,
            pct_esgoto_adequado,
            pct_preta_parda,
            pct_urbana,
            pct_15_24,
        ],
        axis=1,
    ).reset_index()

    df["uf"] = df["municipio"].str.rsplit(" - ", n=1).str[-1]
    return df


FEATURES = [
    "adultos_EM_mais",
    "pct_baixa_renda",
    "pct_internet",
    "pct_esgoto_adequado",
    "pct_preta_parda",
    "pct_urbana",
    "pct_15_24",
]

ALVO = "taxa_sem_EM"


def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove municípios sem alvo/features e duplicatas. Não imputa dado."""
    colunas = ["codigo", "municipio", "uf", "pop_18_24", ALVO, *FEATURES]
    limpo = df[colunas].copy()
    limpo = limpo.dropna(subset=[ALVO, "pop_18_24", *FEATURES])
    limpo = limpo[limpo["pop_18_24"] > 0]
    limpo = limpo.drop_duplicates(subset="codigo")
    limpo["codigo"] = limpo["codigo"].astype(str)
    return limpo.sort_values("codigo").reset_index(drop=True)


def salvar_processado(df: pd.DataFrame, nome: str = "municipios_features.csv") -> Path:
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = DIR_PROCESSED / nome
    df.to_csv(destino, index=False, encoding="utf-8")
    return destino


if __name__ == "__main__":
    resumo = baixar_todas()
    print(resumo[["tabela", "origem", "bytes"]].to_string(index=False))
    base = tratar_dados(construir_features())
    print(f"\nMunicípios: {len(base)}")
    print(base.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print(salvar_processado(base))
