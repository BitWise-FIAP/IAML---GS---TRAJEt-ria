# Roteiro do vídeo — 5 minutos

Formato: gravação de tela com o notebook aberto + narração. Publicar como **não listado** no YouTube.

| Tempo | Bloco | O que mostrar / falar |
|---|---|---|
| 0:00–0:35 | **Problema** | A TRAJEtória doa roupas para jovens em vulnerabilidade. O galpão do piloto fica em São Caetano do Sul, com 40 entregas/mês e R$ 1.000 de logística. A pendência Q03 é: em quais cidades e raio de entrega atuar? |
| 0:35–1:00 | **Por que ML** | O Censo 2022 mostra a taxa de jovens sem ensino médio em cada município, mas não diz onde priorizar. O modelo aprende a relação entre características do território e essa vulnerabilidade, ranqueia e explica. |
| 1:00–1:40 | **Dados** | Notebook §3. Censo 2022 via API SIDRA v3, 7 tabelas, 5.570 municípios, 100% de cobertura, 21,2 milhões de jovens de 18 a 24. Snapshot em `dados/raw/` com SHA-256. Dizer explicitamente: **não** usamos o `seed.json` fictício do app nem a pesquisa de 31 jovens como treino. |
| 1:40–2:20 | **Alvo e features** | Notebook §4 e §5. Alvo = % de jovens 18–24 sem EM completo. Sete features do território. Split 80/20 com semente 42, peso = população jovem, sem vazamento. |
| 2:20–3:10 | **Modelo e busca** | Notebook §6 e §7. Baselines Dummy e Regressão Linear. Random Forest com 24 combinações de `n_estimators`, `max_depth` e `min_samples_leaf`, CV 5-fold **só no treino**. Melhor: 200 árvores, profundidade 12, folha mínima 3. |
| 3:10–3:50 | **Resultados e overfitting** | Notebook §8 e §9. Tabela treino × teste: RF R² 0,826 treino / 0,534 teste; Linear 0,498; Dummy 0. Spearman 0,721. Mostrar a curva de complexidade: a partir de `max_depth=8` o teste para de subir e o treino continua — memorização. |
| 3:50–4:30 | **O que o modelo aprendeu e a resposta da Q03** | Notebook §10 e §11. Importância por permutação: escolaridade dos adultos (0,58), jovens na população (0,19), baixa renda (0,08). Ranking: São Paulo #1 (~291 mil jovens sem EM), Guarulhos #2, Campinas #3, São Bernardo #6, Santo André #9; São Caetano (galpão) em 79º. |
| 4:30–5:00 | **Conclusão e limitações** | O público não está no entorno do galpão: o raio deve cobrir o ABC (São Bernardo, Santo André, Mauá, Diadema) e a escolha da ONG deve seguir a capilaridade nesses territórios. Limitações: dado municipal (falácia ecológica), não causal, Censo é retrato de 2022. |

## Checklist antes de publicar

- [ ] Notebook aberto já executado (Restart & Run All), com as saídas visíveis.
- [ ] Vídeo com no máximo 5 minutos, áudio audível, tela legível (fonte grande).
- [ ] Publicar como **não listado** e copiar o link.
- [ ] Colar o link na seção **Vídeo** do `README.md` e commitar.
