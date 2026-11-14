# TRAJEtória — IA & Machine Learning

**Global Solution 2026.2 · FIAP · Turma 3ESPH**

Modelo de Machine Learning que identifica **onde estão os jovens de 18 a 24 anos
sem ensino médio completo** nos municípios paulistas, usando dados públicos do
**Censo Demográfico 2022 (IBGE)** — para orientar onde o piloto da TRAJEtória deve
atuar.

---

## Ficha da CitizenTech

| Item | Conteúdo |
|---|---|
| CitizenTech | **TRAJEtória** — plataforma social de redistribuição de roupas para jovens atendidos por ONGs |
| Desafio | Conectar cidadania, inclusão e tecnologia a uma experiência acessível |
| Público beneficiado | Jovens em vulnerabilidade, indicados e validados por ONGs parceiras |
| ODS principal | **ODS 8** — Trabalho decente e crescimento econômico |
| ODS secundário | **ODS 10** — Redução das desigualdades |
| Recorte desta disciplina | **ODS 4** — Educação de qualidade (escolaridade jovem) |
| Data | novembro de 2026 |

### Equipe

| Nome completo | RM | Turma | `NomeAluno_RM` | Líder |
|---|---|---|---|---|
| Enrico Ricarte | 558571 | 3ESPH | `EnricoRicarte_RM558571` | sim |
| Pedro Gaspar | 554887 | 3ESPH | `PedroGaspar_RM554887` | não |
| Victor Freire | 556191 | 3ESPH | `VictorFreire_RM556191` | não |

---

## O problema

O piloto da TRAJEtória tem galpão em **São Caetano do Sul**, cerca de **40 entregas
por mês** e orçamento de logística de **R$ 1.000**. Falta a decisão registrada como
pendência **Q03** na documentação funcional: **em quais cidades, bairros e raio de
entrega o piloto deve atuar?**

Se o galpão está no município menos vulnerável do ABC, o raio de entrega precisa ser
desenhado para alcançar quem de fato precisa — e não apenas quem está perto do galpão.

## A pergunta de Machine Learning

> **Quais municípios paulistas concentram mais jovens de 18 a 24 anos sem ensino
> médio completo, e quais características do território explicam essa concentração
> — para priorizar onde o piloto da TRAJEtória deve atuar?**

* **Unidade de análise:** município (`N6`), 5.570 no Brasil e 645 em São Paulo.
* **Alvo:** proporção de jovens de 18 a 24 anos sem ensino médio completo.
* **Features:** escolaridade dos adultos, baixa renda, internet, esgotamento
  sanitário, cor/raça, urbanização e estrutura etária.
* **Modelo:** Random Forest (regressão), com baselines Dummy e Regressão Linear.
* **Validação:** 80/20 com semente fixa; 24 combinações de 3 hiperparâmetros;
  validação cruzada 5-fold **apenas no treino**.

## Resultados principais

| Métrica (teste, 1.114 municípios) | Dummy | Regressão Linear | Random Forest |
|---|---|---|---|
| R² | 0,000 | 0,498 | **0,534** |
| RMSE | 0,1182 | 0,0837 | **0,0807** |
| MAE | 0,0968 | 0,0665 | **0,0629** |
| Spearman | — | 0,696 | **0,721** |

* Melhor configuração: `n_estimators=200`, `max_depth=12`, `min_samples_leaf=3`
  (R² de validação cruzada ≈ 0,541, RMSE ≈ 0,078).
* **Gap treino-teste:** R² 0,826 no treino contra 0,534 no teste. A curva de
  complexidade mostra que a partir de `max_depth=8` o teste para de melhorar e o
  treino continua subindo — o limite de profundidade foi o que conteve a memorização.
* **Variáveis mais importantes** (permutação, teste): escolaridade dos adultos
  (0,583), proporção de jovens de 15 a 24 anos (0,195) e baixa renda (0,083).

### O que isso significa para o piloto

* **São Paulo** concentra o maior volume absoluto: 1.129.866 jovens de 18 a 24 anos,
  dos quais ~291 mil sem ensino médio completo.
* **São Bernardo do Campo** e **Santo André** estão entre os **dez municípios
  paulistas com maior volume** de jovens sem EM (17.910 e 15.392).
* **São Caetano do Sul**, onde está o galpão, é o **menos vulnerável do ABC**
  (14,2%) e ocupa a **79ª posição** estadual em volume.

Conclusão operacional: o público da TRAJEtória **não está no entorno imediato do
galpão**. O raio de entrega e a escolha da ONG parceira precisam considerar os
municípios vizinhos de maior volume — São Bernardo, Santo André, Mauá e Diadema.

## Estrutura do repositório

```
.
├── README.md                  # este arquivo
├── RELATORIO_ML.md            # relatório de 2 a 4 páginas
├── .gitignore
└── ml/
    ├── trajetoria_ml.ipynb    # notebook EXECUTADO (Restart & Run All)
    ├── requirements.txt
    ├── src/
    │   └── ibge_api.py        # coleta e tratamento da API do IBGE
    ├── dados/
    │   ├── README.md          # fonte, URLs, dicionário, licença, hashes
    │   ├── raw/               # snapshot das respostas do IBGE (JSON)
    │   └── processed/
    │       ├── municipios_features.csv
    │       └── prioridade_sp.csv
    └── figuras/               # gráficos gerados pelo notebook
```

## Como reproduzir

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

jupyter lab trajetoria_ml.ipynb   # abrir e ver as saídas salvas
```

Para reexecutar do zero (Restart & Run All) e regravar as saídas:

```bash
jupyter nbconvert --to notebook --execute --inplace trajetoria_ml.ipynb
```

O notebook funciona **offline**: o snapshot do IBGE está em `ml/dados/raw/`.
Para forçar nova coleta, rode `ibge_api.baixar_todas(forcar=True)`.

## Uso de dados

| Fonte | Uso |
|---|---|
| Censo 2022 / IBGE (API SIDRA v3) | **Base de treino e teste** — dado público, municipal |
| Pesquisa com 31 jovens (projeto) | **Motivação** apenas — amostra pequena e não pública |
| `seed.json` do aplicativo | **Não utilizado** — dados fictícios, recusados pela rubrica |

Detalhes de cada tabela, URLs, período, dicionário e SHA-256: **[`ml/dados/README.md`](ml/dados/README.md)**.

## Entregáveis da disciplina

| Critério | Onde está |
|---|---|
| Problema e dados reais (2,0) | `ml/trajetoria_ml.ipynb` §§1–3, `ml/dados/README.md` |
| Escolha do modelo (1,0) | §7 do notebook e §4 do relatório |
| Preparação e split (2,0) | §5 do notebook |
| Treino com ≥10 combinações (2,0) | §7 — 24 combinações, CV só no treino |
| Treino vs teste e overfitting (2,0) | §§8–9 e figuras `03`, `04` |
| Conclusão para o cidadão (1,0) | §§11–12 e `ml/dados/processed/prioridade_sp.csv` |

## Limitações

* Dado **agregado por município**: não permite inferência sobre indivíduos
  (falácia ecológica) e não estabelece causalidade.
* Censo 2022 é um **retrato**, não uma previsão.
* Municípios pequenos têm taxas instáveis; o modelo suaviza os extremos (São
  Caetano do Sul: 14,2% observado contra 20,1% previsto). Por isso o CSV operacional
  mantém a coluna observada ao lado da prevista.

## Relatório

A discussão completa — problema, dados, metodologia, overfitting, leitura para o
cidadão e limitações — está em **[`RELATORIO_ML.md`](RELATORIO_ML.md)**.

## Vídeo

Apresentação de 5 minutos: **[preencher com o link do YouTube não listado]**.
Roteiro de gravação em [`ROTEIRO_VIDEO.md`](ROTEIRO_VIDEO.md).
