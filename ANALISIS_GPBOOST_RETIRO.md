# Por qué se retiró GPBoost: la data no tiene estructura para efectos mixtos

**Decisión (2026-06-22): se retira el backend GPBoost del pipeline.** Queda
LGB + XGB. Este documento registra la evidencia — toda a nivel de **datos**,
no de código — que la respalda.

## Resumen

GPBoost = árboles LightGBM + efectos aleatorios (modelo mixto). Solo aporta
sobre LGB si la estructura de grupo (FUNDO/FORMATO) lleva señal que los árboles
no capturan. **En esta data no la lleva.** Tres pruebas independientes
convergen:

## 1. La señal de grupo se evapora tras los lags (cascada ICC, POP)

ICC = % de varianza del target que vive ENTRE grupos.

| Nivel | FUNDO | FORMATO | FUNDO×FORMATO |
|---|---|---|---|
| Marginal (target crudo) | 12.9% | 22.4% | 24.5% |
| Tras features (sin grupos) | 2.3% | 17.4% | 10.7% |
| **Tras features + lag-por-grupo** | **0.0%** | **0.3%** | **0.2%** |

Los lags que el pipeline ya computa por FUNDO×FORMATO **absorben toda** la
estructura de grupo que un intercepto aleatorio modelaría. El RE queda
redundante.

## 2. El GPBoost registrado "usa" el grupo solo porque SUBAJUSTÓ

`cov_pars` del modelo POP registrado (espacio log1p):
`Error_var=0.0116 | FUNDO=0.061 | FORMATO=0.0037`. La varianza de FUNDO parece
grande — pero el modelo usó **~65 árboles** (`[63,68,76,62,64]`) frente a los
**~1400 de LGB** (`[1409,649,1992,1323,1994]`). Cortó tempranísimo (early
stopping por NLL del GP) → los árboles no aprendieron los lags → el RE absorbió
la señal que los árboles dejaron. El `cov_par` alto es un **artefacto del
subajuste**, no señal irreducible.

## 3. No existe agrupamiento con (alta cardinalidad + señal residual)

Los efectos aleatorios brillan con muchos grupos chicos que lleven señal no
capturada por los features. Se probaron todos los agrupamientos disponibles
(ICC residual tras features+lag, POP):

| Grupo | niveles | ICC residual | ¿RE útil? |
|---|---|---|---|
| COSECHADOR | 3567 | 0.0% | No (alta card, señal nula) |
| CALIBRE | 9 | 11.2% | No (señal pero baja card → efecto fijo basta) |
| TIPO DE COSECHA | 6 | 7.8% | No (idem) |
| FUNDO / FORMATO | 3 / 10 | ~0% / ~7% | No (redundante con lags) |

**Cross-variedad** (no solo POP): ROSITA, SEKOYA POP ORGÁNICA y MÁGICA retienen
ICC residual ~5-9%, pero siempre de FORMATO/FUNDO de **baja cardinalidad** —
donde un efecto fijo (one-hot, que LGB ya usa) rinde igual que uno aleatorio.

## 4. La heterocedasticidad existe pero GPB no la modela

A9 (std 1.75) y GRANEL (std 1.67) son más dispersos que el resto — hay
heterocedasticidad real por grupo. Pero la likelihood **gaussiana** de GPBoost
es **homoscedástica** (un solo `Error_var`): sus interceptos aleatorios mueven
la MEDIA por grupo, no la VARIANZA. La estructura que existe es del tipo
equivocado para su mecanismo.

## Conclusión y qué se hizo en su lugar

GPBoost no podía ganar el campeonato en NINGUNA variedad analizada (perdía 16.1%
vs 14.36% de LGB) y costaba ~60% del wall-time. Se retira por completo.

La señal real que SÍ existe se dejó disponible como **features opt-in** (ayudan
a LGB y XGB por igual, ver `src/config.py`):
- `ENABLE_EXTRA_CATEGORICALS`: CALIBRE + TIPO DE COSECHA (A/B: −1.58 pp de MAPE).
  Requiere que el TRAINING_FILE canónico traiga esas columnas.
- `ENABLE_TARGET_VOLATILITY`: dispersión del target por grupo, el modo en que un
  árbol aprovecha la heterocedasticidad (A/B: −0.41 pp).
- La incertidumbre por grupo ya se modela en las bandas conformales por fundo
  del API (`q_by_fundo`).

> **Importante:** ambos flags están OFF por defecto. Los A/B se midieron sobre
> un modelo proxy, no el pipeline de producción. Como tocan el feature set del
> campeón (LGB), NO se adoptan como default sin validar en prod_xl que MEJORAN
> el 14.36% actual — analizar es para mejorar, no empeorar. El baseline (flags
> OFF) reproduce el campeón actual bit-idéntico.
