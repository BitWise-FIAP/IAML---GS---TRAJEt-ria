# Figuras

Todas geradas pelo notebook (`trajetoria_ml.ipynb`) em 160 dpi, a partir do snapshot do Censo 2022.

| Arquivo | Seção | O que mostra | Por que importa |
|---|---|---|---|
| `01_distribuicao_alvo.png` | §4 | Histograma da taxa de jovens 18–24 sem EM (Brasil × São Paulo) e boxplot Brasil × SP × ABC | Mostra que o ABC não é homogêneo: a média esconde realidades muito diferentes |
| `02_correlacao.png` | §4 | Matriz de correlação entre alvo e as 7 features | Antecipa o sinal (escolaridade adulta −0,58; baixa renda +0,57) e checa multicolinearidade |
| `03_curva_complexidade.png` | §9 | R² de treino, validação cruzada e teste por `max_depth` | Evidencia o overfitting: a partir de `max_depth=8` o teste estagna e o treino continua subindo |
| `04_previsto_vs_observado.png` | §8 | Dispersão previsto × observado no teste (por região) e distribuição dos resíduos | Mostra onde o modelo erra e que os resíduos são aproximadamente centrados em zero |
| `05_importancia_permutacao.png` | §10 | Queda média no R² ao embaralhar cada feature (teste) | Sustenta a explicação: escolaridade dos adultos é o fator dominante |
| `06_volume_vs_intensidade.png` | §11 | População 18–24 (log) × taxa prevista, com tamanho pelo volume e destaques do ABC | É o gráfico que responde à Q03: onde há gente e onde há intensidade |

Reproduzir:

```bash
jupyter nbconvert --to notebook --execute --inplace trajetoria_ml.ipynb
```
