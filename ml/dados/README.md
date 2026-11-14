# Dados — TRAJEtória / IA & Machine Learning

Esta pasta contém **apenas dados públicos do IBGE**. Não há dado pessoal, dado de
cliente, nem dado fictício do aplicativo.

## 1. Fonte

| Item | Conteúdo |
|---|---|
| Instituição | IBGE — Instituto Brasileiro de Geografia e Estatística |
| Pesquisa | Censo Demográfico 2022 |
| Período | 2022 (retrato, não série histórica) |
| Nível territorial | Município (`N6`) — todos os 5.570 municípios do Brasil |
| Acesso | API v3 do SIDRA (agregados), sem download manual |
| URL base | `https://servicodados.ibge.gov.br/api/v3/agregados` |
| Data da coleta | 14/11/2026 |
| Licença | Dados públicos (Lei de Acesso à Informação, Lei 12.527/2011). Uso livre com citação da fonte. |

Referência para citação:

> IBGE. **Censo Demográfico 2022**: resultados do universo por município. Rio de
> Janeiro: IBGE, 2023. Disponível em: https://sidra.ibge.gov.br. Acesso em: 14 nov. 2026.

## 2. O que **não** foi usado (e por quê)

* **`assets/mock/seed.json` do aplicativo.** O próprio README do app declara que
  peças, jovens e pedidos são fictícios (`"Bairro Fictício"`, nomes inventados).
  A rubrica da disciplina não aceita dado simulado.
* **A pesquisa com os 31 jovens** (`TRAJEtoria_Resultados_Pesquisa_Jovens.pdf`).
  É real, mas é uma amostra pequena, não pública e sem representatividade
  territorial. Entra no relatório apenas como **motivação**.

## 3. Arquivos brutos (`raw/`)

Cada arquivo é a resposta exata da API, sem edição, acompanhado de um `.meta.json`
com URL, data da coleta e SHA-256.

| Tabela | Variável | Classificação | Descrição | Linhas | Municípios | Tamanho | SHA-256 (início) |
|---|---|---|---|---|---|---|---|
| 10061 | 2667 | `58[100052,108866]` `1568[all]` `2[6794]` `86[95251]` | Pessoas de 18+ por nível de instrução e grupo de idade | 55.700 | 5.570 | 7,6 MB | `e58d3567a67f` |
| 10296 | 13604 | `2[6794]` `86[95251]` `386[all]` | Rendimento nominal mensal domiciliar per capita | 66.840 | 5.570 | 9,0 MB | `6fe8c5f072a3` |
| 9936 | 381 | `2072[all]` `63[95826]` `125[2932]` | Domicílios por existência de conexão à internet | 16.710 | 5.570 | 2,3 MB | `7b89a7ecda45` |
| 6805 | 381 | `11558[all]` | Domicílios por tipo de esgotamento sanitário | 55.700 | 5.570 | 7,5 MB | `7438aef77453` |
| 9605 | 93 | `86[all]` | População residente por cor ou raça | 33.420 | 5.570 | 4,5 MB | `b0e99856949f` |
| 9923 | 93 | `1[all]` | População residente por situação do domicílio | 16.710 | 5.570 | 2,3 MB | `9dd20749e8f5` |
| 9514 | 93 | `2[6794]` `287[93086,93087,100362]` `286[113635]` | População residente por sexo e idade | 16.710 | 5.570 | 2,3 MB | `d3e7cc3b8953` |

Cada `*.meta.json` traz o SHA-256 completo, a URL exata e o horário UTC da coleta.

## 4. Como reproduzir a coleta

```bash
# dentro de ml/
python src/ibge_api.py                 # baixa o que faltar e gera os CSVs
```

Ou, a partir do notebook:

```python
from src import ibge_api

ibge_api.baixar_todas()                # usa o snapshot local (offline)
ibge_api.baixar_todas(forcar=True)     # força nova coleta na API do IBGE
```

Sem internet, o notebook roda normalmente: o snapshot em `raw/` é a fonte.

## 5. Dicionário — `processed/municipios_features.csv`

Uma linha por município (5.570), usado no treino e no teste.

| Coluna | Descrição | Origem |
|---|---|---|
| `codigo` | Código IBGE do município (7 dígitos) | todas |
| `municipio` | Nome do município com UF (`"São Paulo - SP"`) | todas |
| `uf` | Sigla da unidade da federação | derivada do nome |
| `pop_18_24` | População de 18 a 24 anos (peso amostral e volume) | 10061 |
| `taxa_sem_EM` | **Alvo.** Proporção de jovens de 18 a 24 anos sem ensino médio completo | 10061 |
| `adultos_EM_mais` | Proporção de adultos (25+) com ensino médio completo ou mais | 10061 |
| `pct_baixa_renda` | Proporção de pessoas sem rendimento ou com até 1/2 salário mínimo | 10296 |
| `pct_internet` | Proporção de domicílios com conexão à internet | 9936 |
| `pct_esgoto_adequado` | Proporção de domicílios com rede geral/rede pluvial/fossa ligada à rede ou fossa séptica | 6805 |
| `pct_preta_parda` | Proporção de população preta ou parda | 9605 |
| `pct_urbana` | Proporção de população em situação urbana | 9923 |
| `pct_15_24` | Proporção de jovens de 15 a 24 anos na população total | 9514 |

**Definição do alvo (tabela 10061):**

```
taxa_sem_EM = ("Sem instrução e fundamental incompleto" + "Fundamental completo e médio incompleto")
              / Total de pessoas de 18 a 24 anos
```

## 6. Dicionário — `processed/prioridade_sp.csv`

Uma linha por município paulista (645), produzida pelo notebook.

| Coluna | Descrição |
|---|---|
| `codigo_ibge` | Código IBGE |
| `municipio` | Nome do município |
| `regiao` | `Capital`, `ABC paulista` ou `Demais municípios de SP` |
| `pop_18_24` | População de 18 a 24 anos |
| `taxa_observada` | Taxa medida no Censo 2022 (verdade de campo) |
| `taxa_prevista` | Taxa estimada pelo Random Forest |
| `jovens_sem_EM_estimado` | `taxa_prevista × pop_18_24` — volume estimado |
| `posicao_volume` | Posição no estado por volume estimado (1 = maior) |
| `prioridade` | Percentil do volume estimado (0 a 100) |
| `nivel_prioridade` | `Alta` (top 10% em volume), `Média` ou `Baixa` |

## 7. Tratamento e limitações

* `...` (não disponível) e `X` (sigilo estatístico) viram `NaN`; `-` vira `0`.
* Municípios sem alvo, sem população de 18 a 24 anos ou sem alguma feature foram
  **removidos**, nunca imputados. Nenhum município foi removido: cobertura de 100%.
* Percentuais sempre no intervalo [0, 1].
* O dado é **agregado por município**. Não permite inferência sobre indivíduos
  (falácia ecológica) e não estabelece causalidade.
* O Censo 2022 é um retrato. O modelo não prevê o futuro.
