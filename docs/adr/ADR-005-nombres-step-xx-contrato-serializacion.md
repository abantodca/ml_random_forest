# ADR-005 — Los nombres `step_XX_verbo/` son un contrato de serialización

**Estado:** accepted
**Número asignado:** 2026-07-20, en la reorganización documental. La decisión y su rationale ya
existían (`README.md:419-424`, `CLAUDE.md:107-108`, `CONTRIBUTING.md:57-58`); lo que no existía era
el número.

## Contexto

Los módulos del pipeline se llaman `step_01_load`, `step_02_clean`, `step_03_features`,
`step_04_train`, `step_05_evaluate`, `step_06_track`. Es un naming inusual —normalmente uno llamaría
a esas carpetas `loading/`, `cleaning/`, `features/`— y periódicamente alguien propone "limpiarlo".

`README.md:419-424` da los tres motivos por los que no:

> "Los módulos `step_XX_verbo/` codifican el orden del pipeline en el propio nombre — el lector
> entiende la secuencia sin abrir un diagrama. Se mantienen así por:
>  1. **Determinismo visual**: `01_load → 02_clean → 03_features → 04_train → 05_evaluate → 06_track`
>     es legible sin contexto.
>  2. **Compatibilidad Python**: módulos no pueden empezar con dígitos puros (`01_load` falla); el
>     prefijo `step_` lo resuelve.
>  3. **Estabilidad de imports**: renombrar implica tocar todos los `from src.step_X import ...` y los
>     `.joblib` ya serializados (que recuerdan el path del pipeline). **El costo supera al
>     beneficio.**"

El tercer punto es el que convierte esto en una decisión arquitectónica y no en una preferencia de
estilo. Los transformers custom se picklean **con su path de import dentro**:

> "La imagen de la API necesita el paquete `src/` raiz para des-picklear los modelos de MLflow (los
> transformers `LagFeatureTransformer`, `FeatureGenerator`, etc. **se serializan con rutas
> `src.step_03_features.*`**). Por eso el **contexto de build es la raiz**."

## Decisión

Los nombres de los módulos `step_XX_verbo/` **no se renombran**, y los símbolos públicos **no cambian
de path**. Un refactor solo puede mover privados y helpers, re-exportándolos desde el módulo o el
`__init__.py` que ya importa el call-site.

## Consecuencias

**Se gana:**

- Los `.joblib` viejos siguen deserializando. Un modelo entrenado hace seis meses se puede cargar hoy.
- La API y el trainer comparten el mismo `src/`, y por eso el `api/Dockerfile` tiene la **raíz del
  repo** como contexto de build. Eso también está protegido por esta decisión.

**Se pierde:**

- El naming queda "raro" a perpetuidad para quien llega nuevo. Es el precio de la compatibilidad
  hacia atrás de los artefactos.
- Cualquier reorganización de `src/` es más cara de lo que parece.

**Regla operativa derivada:** el refactor de `step_03_features/lag_features.py` está especificado
como *"cómputo puro → `_lag_compute.py`; **`LagFeatureTransformer` NO se mueve** (su path está
horneado en `.joblib`)"*, con riesgo **alto** y condicionado a un test de pickle verde antes y
después. Ver [ADR-008](ADR-008-ci-sin-tests-todavia.md): ese test ya no existe, así que ese refactor
hoy **no es ejecutable como está especificado**.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Renombrar a `loading/`, `features/`, … | Rompe todos los `.joblib` serializados y todos los imports. *"El costo supera al beneficio"* |
| Prefijo numérico puro (`01_load`) | Python no permite módulos que empiecen con dígito |
| Vendorizar una copia de `src/` dentro de `api/` | Deriva silenciosa entre el código que entrena y el que sirve (`CLAUDE.md:93-95`) |

## Dónde vive en el código

- `src/step_01_load/` … `src/step_06_track/` — la estructura entera
- `api/Dockerfile` — contexto de build = raíz del repo, para poder `COPY src/`
- `src/step_03_features/lag_features.py::LagFeatureTransformer` — el símbolo cuyo path está horneado
- `docs/REFACTOR_ARQUITECTURA_2026-06-23.md` — tabla de qué se puede y qué no se puede mover
