# Relatório — IA & Machine Learning

## TRAJEtória: onde estão os jovens de 18 a 24 anos sem ensino médio completo?

**Global Solution 2026.2 · FIAP · Turma 3ESPH · novembro de 2026**

| Integrante | RM | Turma |
|---|---|---|
| Enrico Ricarte (líder) | 558571 | 3ESPH |
| Pedro Gaspar | 554887 | 3ESPH |
| Victor Freire | 556191 | 3ESPH |

**CitizenTech:** TRAJEtória — plataforma social de redistribuição de roupas para
jovens atendidos por ONGs.
**ODS:** 8 (principal), 10 (secundário) e 4 (recorte desta disciplina).

---

## 1. Resumo executivo

A TRAJEtória vai operar um piloto com galpão em **São Caetano do Sul**, cerca de
**40 entregas por mês** e **R$ 1.000** de orçamento de logística. A documentação
funcional deixa em aberto a decisão: em quais cidades, bairros e raio de
entrega o piloto deve atuar.

Este trabalho responde à pergunta com dados públicos do **Censo Demográfico 2022
(IBGE)**, cobrindo os **5.570 municípios brasileiros**. Um **Random Forest** foi
treinado para estimar a proporção de jovens de 18 a 24 anos **sem ensino médio
completo** a partir de sete características do território (escolaridade dos adultos,
baixa renda, internet, esgotamento sanitário, cor/raça, urbanização e estrutura
etária).

**Resultado principal:** o modelo explica 53,4% da variância no conjunto de teste
(R² = 0,534), contra 49,8% da regressão linear e 0% da média — com RMSE de 8,1 pontos
percentuais e Spearman de 0,721. A leitura para o piloto: o galpão está no município
menos vulnerável do ABC; os jovens que o projeto quer atender estão, sobretudo, em
**São Bernardo do Campo, Santo André, Mauá e Diadema**, além da própria capital.

---

## 2. Problema e decisão

A TRAJEtória conecta jovens em vulnerabilidade a roupas sociais doadas, com curadoria
e entrega em casa. A pesquisa com 31 jovens do público-alvo mostrou que 61,3% não
sabem o que é "adequado vestir" e que 54,8% já deixaram de ir a uma entrevista por
causa da roupa. A dor é real — mas a pesquisa não diz **onde** essas pessoas estão.

Sem essa resposta, o piloto corre o risco de gastar seu orçamento de entrega
atendendo quem mora ao lado do galpão, e não quem mais precisa. A decisão Q03 é,
portanto, uma decisão de priorização territorial — e é exatamente o tipo de problema
em que um modelo supervisionado ajuda: aprende a relação entre características do
território e vulnerabilidade educacional, ranqueia e explica.

**Pergunta de ML:** *quais municípios paulistas concentram mais jovens de 18 a 24
anos sem ensino médio completo, e quais características do território explicam essa
concentração?*

**Critério de sucesso definido antes do treino:** superar com folga os baselines no
teste, manter alta a ordenação dos territórios (Spearman) e conseguir explicar os
fatores em linguagem acessível.

---

## 3. Dados

### 3.1 Fonte e recorte

Todas as variáveis vêm do **Censo Demográfico 2022**, agregadas por município, obtidas pela **API v3 do SIDRA** do IBGE. 

| Tabela | O que fornece |
|---|---|
| 10061 | **Alvo** (% de jovens 18–24 sem EM) e escolaridade dos adultos |
| 10296 | Percentual de pessoas em domicílios de baixa renda |
| 9936 | Percentual de domicílios com internet |
| 6805 | Percentual de domicílios com esgotamento adequado |
| 9605 | Percentual de população preta ou parda |
| 9923 | Percentual de população urbana |
| 9514 | Percentual de jovens de 15 a 24 anos na população |

**Cobertura:** 5.570 municípios, com as sete tabelas completas — nenhuma imputação
foi necessária. São 21,2 milhões de jovens de 18 a 24 anos na base.

### 3.2 Tratamento

`...` (não disponível) e `X` (sigilo) foram convertidos em `NaN`; `-` virou zero.
Municípios sem alvo, sem população de 18 a 24 anos ou sem alguma feature seriam
removidos — e não imputados. Nenhum foi. As proporções foram calculadas sempre com
o denominador correto de cada tabela (pessoas, domicílios ou população total), e
não há duplicidade de código IBGE.

---

## 4. Metodologia

### 4.1 Variável-alvo e features

O alvo é a proporção de jovens de 18 a 24 anos **sem ensino médio completo**:

```
taxa_sem_EM = (sem instrução + fundamental incompleto
               + fundamental completo e médio incompleto) / total de 18 a 24 anos
```

As sete features são percentuais do território. A escolaridade dos adultos usa a
faixa **25 anos ou mais**, portanto não se sobrepõe ao alvo (18 a 24). Não há
vazamento de dados: nenhuma feature é derivada do alvo nem de outro recorte da
mesma distribuição.

### 4.2 Split, pesos e validação

* **Split 80/20** com `random_state=42`: 4.456 municípios para treino, 1.114 para
  teste. O teste permaneceu intocado até a avaliação final.
* **Peso amostral** = população de 18 a 24 anos. Municípios minúsculos geram taxas
  instáveis (um município com 56 jovens oscila dezenas de pontos percentuais com
  poucas pessoas); sem peso, o modelo persegue ruído.
* **Validação cruzada 5-fold apenas no treino** (`KFold`, `shuffle=True`,
  `random_state=42`). Usar o teste na CV tornaria o R² de teste otimista.

### 4.3 Modelos e busca de hiperparâmetros

Três modelos, do mais simples ao mais complexo:

1. **Dummy (média do treino)** — prova que existe sinal.
2. **Regressão Linear** com padronização — baseline honesto.
3. **Random Forest** — escolhido por capturar não linearidades e interações
   (renda × escolaridade × saneamento) sem engenharia manual, ser robusto a
   outliers e escalas, e oferecer importância de variáveis para a explicação.

A busca em grade testou **24 combinações** de três hiperparâmetros, com critério de
menor RMSE médio na validação:

| Hiperparâmetro | Valores | Papel |
|---|---|---|
| `n_estimators` | 200, 400 | estabilidade da média das árvores |
| `max_depth` | 4, 8, 12, sem limite | complexidade / risco de overfitting |
| `min_samples_leaf` | 1, 3, 5 | suavização das folhas |

**Métricas:** RMSE e MAE (erro na proporção), R² (variância explicada) e Spearman
(ordenação — decisiva, porque o produto final é um ranking). Também foram calculadas
métricas **ponderadas** pela população jovem.

---

## 5. Resultados

### 5.1 Treino vs teste

| Modelo | Conjunto | RMSE | MAE | R² | Spearman |
|---|---|---|---|---|---|
| Dummy (média) | treino | 0,1149 | 0,0929 | 0,000 | — |
| Dummy (média) | teste | 0,1182 | 0,0968 | 0,000 | — |
| Regressão Linear | treino | 0,0810 | 0,0645 | 0,503 | 0,704 |
| Regressão Linear | teste | 0,0837 | 0,0665 | 0,498 | 0,696 |
| **Random Forest** | **treino** | **0,0479** | **0,0373** | **0,826** | **0,914** |
| **Random Forest** | **teste** | **0,0807** | **0,0629** | **0,534** | **0,721** |

O Random Forest supera a regressão linear em R² (+0,035), reduz o RMSE em 3,6% e
melhora a ordenação (Spearman +0,025). Ponderando pela população jovem, o R² de
teste sobe para **0,674** — o modelo acerta mais nos municípios que concentram
gente, que são justamente os que importam para a operação.

A melhor configuração foi `n_estimators=200`, `max_depth=12`,
`min_samples_leaf=3`, com R² de validação cruzada de **0,541** e desvio entre folds
de apenas 0,0016 — resultado estável, não sorte de um fold.

### 5.2 Overfitting

O gap é explícito: **R² 0,826 no treino contra 0,534 no teste**. A curva de
complexidade (figura `03`) mostra o mecanismo:

| `max_depth` | R² treino | R² validação | R² teste |
|---|---|---|---|
| 2 | 0,418 | 0,407 | 0,387 |
| 4 | 0,521 | 0,490 | 0,468 |
| 8 | 0,691 | 0,537 | 0,527 |
| 12 | 0,826 | 0,540 | 0,534 |
| sem limite | 0,870 | 0,539 | 0,534 |

A partir de `max_depth=8` o teste praticamente para de melhorar (0,527 → 0,534),
enquanto o treino continua subindo até 0,870. Ou seja: profundidade adicional só
produz **memorização**. A escolha do grid (profundidade limitada e
`min_samples_leaf=3`) foi o que conteve o overfitting; o desvio-padrão baixo entre
folds confirma que o resultado não é instável.

Esse gap também tem uma explicação substantiva: parte da variabilidade municipal
não é explicável por indicadores estruturais — depende de história local, oferta de
escola, política educacional e migração. Um R² de teste em torno de 0,5 é o teto
realista para sete variáveis agregadas.

### 5.3 O que explica a vulnerabilidade

Importância por permutação no teste (queda média no R² ao embaralhar a variável):

| Feature | Importância | Desvio |
|---|---|---|
| `adultos_EM_mais` | 0,583 | 0,027 |
| `pct_15_24` | 0,195 | 0,019 |
| `pct_baixa_renda` | 0,083 | 0,008 |
| `pct_urbana` | 0,052 | 0,006 |
| `pct_preta_parda` | 0,044 | 0,005 |
| `pct_internet` | 0,027 | 0,005 |
| `pct_esgoto_adequado` | 0,009 | 0,003 |

A escolaridade dos adultos é, de longe, o fator mais associado — o que é coerente
com a literatura de transmissão intergeracional da educação. Municípios com
população mais jovem e mais pobre tendem a ter mais jovens sem ensino médio. É
importante não ler isso como causalidade: são associações estruturais.

---

## 6. Leitura para o cidadão

O modelo foi aplicado aos **645 municípios paulistas**. A prioridade é o
**percentil do volume estimado** de jovens sem ensino médio (`taxa prevista ×
população de 18 a 24 anos`), porque o piloto tem 40 entregas por mês: o que
dimensiona a operação é o número de jovens, não a taxa.

| Posição | Município | Jovens 18–24 | Taxa observada | Jovens sem EM (est.) |
|---|---|---|---|---|
| 1 | São Paulo | 1.129.866 | 26,4% | 291.410 |
| 2 | Guarulhos | 137.636 | 26,6% | 36.032 |
| 3 | Campinas | 108.202 | 25,0% | 25.706 |
| 6 | **São Bernardo do Campo** | 77.463 | 23,4% | 17.910 |
| 9 | **Santo André** | 67.078 | 20,4% | 15.392 |
| — | **Mauá** | 43.807 | 22,1% | 11.255 |
| — | **Diadema** | 42.340 | 20,9% | 10.407 |
| 79 | **São Caetano do Sul** (galpão) | 12.353 | 14,2% | 2.482 |

**O que isso diz ao piloto.** O galpão está em São Caetano do Sul, o município
menos vulnerável do ABC. O público que a TRAJEtória quer atender não está no
entorno imediato: está nos vizinhos. O ABC reúne 258.852 jovens de 18 a 24 anos,
dos quais cerca de **61 mil sem ensino médio completo** — e quase 90% desse
contingente está em São Bernardo, Santo André, Mauá e Diadema.

Três recomendações concretas:

1. **Raio de entrega:** desenhar o piloto para o ABC, com prioridade em São
   Bernardo e Santo André, e não restringi-lo ao município do galpão.
2. **Parceria:** escolher a ONG pela capilaridade nesses territórios, não pela
   proximidade do galpão.
3. **Comunicação:** o projeto pode afirmar, com fonte verificável, onde estão os
   jovens que quer atender — o que fortalece a captação de patrocínio.

A capital é um caso à parte: é o maior volume absoluto do estado (291 mil jovens
sem EM), mas tem taxa mediana. Para um piloto de bairro, o critério deve ser a taxa
dentro do território escolhido; para dimensionar a operação, o volume.

---

## 7. Limitações

* **Falácia ecológica.** O dado é municipal; o modelo fala do território, não do
  indivíduo. Não se pode concluir que todo jovem de um município vulnerável precisa
  do serviço.
* **Não é causal.** As associações não provam que renda ou saneamento causem evasão
  escolar. Servem para priorizar e levantar hipóteses.
* **Censo 2022 é um retrato.** Não antecipa mudanças econômicas ou demográficas.
* **Extremos suavizados.** O modelo puxa valores extremos para o meio: São Caetano
  do Sul tem 14,2% observado e 20,1% previsto. Por isso o CSV operacional mantém a
  coluna observada ao lado da prevista.
* **Municípios pequenos.** Taxas instáveis; o peso populacional e a priorização por
  volume mitigam, mas não eliminam o problema.
* **O alvo já é publicado.** O valor do modelo não está em "descobrir" o Censo, e
  sim em explicar os fatores, ranquear com critério transparente e aplicar o score
  a recortes onde o indicador não está disponível.
---

## Referências
IBGE. **Censo Demográfico 2022**: resultados do universo por município. Rio de
Janeiro: IBGE, 2023. Disponível em: https://sidra.ibge.gov.br. Acesso em: 14 nov. 2026.

BREIMAN, L. Random Forests. **Machine Learning**, v. 45, p. 5–32, 2001.

PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. **Journal of Machine
Learning Research**, v. 12, p. 2825–2830, 2011.
