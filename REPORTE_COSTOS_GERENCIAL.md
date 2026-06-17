# Reporte de Costos — Plataforma MLOps `ml-training` (AWS)

**Para:** Gerencia
**Asunto:** Costo mensual de la plataforma de pronósticos según horario de encendido
**Región:** AWS `us-east-1` · **Moneda:** USD · **Fecha:** 2026-05-26

---

## 1. Resumen ejecutivo

La plataforma (entrenamiento de modelos + MLflow + API de pronósticos + dashboard) puede operar en distintos horarios mediante un **apagado automático programado** (scheduler) que detiene la base de datos y los servicios fuera del horario de uso. El **balanceador (ALB) y el NAT Gateway permanecen activos 24/7** y constituyen un **costo fijo inevitable de ~$58/mes**, independiente del horario.

| Escenario | Encendido | Horas/mes | **Costo mensual** | Costo anual | Ahorro vs 24/7 |
|---|---|---:|---:|---:|---:|
| **A — 24/7** (siempre prendido) | 24 h × 7 días | **730 h** | **≈ $226** | ≈ $2,712 | — |
| **B — 6 días/sem × 8 h** | 48 h/semana | **208 h** | **≈ $106** | ≈ $1,272 | **53 %** |
| **C — 3 días/sem × 4 h** *(configuración actual)* | 12 h/semana | **52 h** | **≈ $70** | ≈ $840 | **69 %** |

> **Recomendación:** la configuración actual (**Escenario C**, ~$70/mes) cubre la operación normal con un ahorro del **69 %** frente a tener todo prendido 24/7. Subir a **Escenario B** (~$106/mes) solo se justifica si se requiere disponibilidad la mayor parte de la semana laboral.

---

## 2. ¿Por qué llevar esto a la nube? (en lenguaje no técnico)

Esta sección explica, sin tecnicismos, **qué problema resolvemos** al mover la plataforma a la nube y **por qué el costo anterior es una inversión y no un gasto**.

### 2.1 El punto de partida: un modelo en la computadora de una persona

Hoy un pronóstico de demanda típicamente nace así: un analista entrena un modelo en **su propia laptop**, guarda el resultado en un archivo y lo comparte por correo o en una carpeta. Funciona… hasta que aparecen los problemas del día a día:

- **Si esa persona no está** (vacaciones, se enferma, renuncia), nadie más sabe cómo rehacer el pronóstico.
- **No hay forma de saber si el modelo sigue siendo bueno.** Los hábitos de compra cambian; un modelo que acertaba hace 6 meses puede estar fallando hoy y nadie se entera.
- **No se puede repetir el resultado.** Si alguien pregunta “¿cómo llegaste a este número?”, reconstruirlo es difícil o imposible.
- **No escala.** Una laptop entrena un producto a la vez; atender decenas de productos o varias tiendas se vuelve lento y manual.
- **El acceso es frágil.** El pronóstico vive en un archivo, no en un servicio al que el negocio pueda consultar cuando lo necesita.

En resumen: el conocimiento queda **atrapado en una persona y en una máquina**, sin respaldo, sin control de calidad y sin continuidad.

### 2.2 Qué es “MLOps” (la idea en una frase)

**MLOps es aplicar al ciclo de los modelos de datos las mismas buenas prácticas que ya usamos para operar cualquier proceso serio del negocio:** que esté documentado, que se pueda repetir, que alguien lo supervise y que no dependa de una sola persona.

Una analogía simple: pasamos de **“un cocinero que tiene la receta en la cabeza”** a **“una cocina con la receta escrita, ingredientes estandarizados, control de calidad y un local abierto en horario fijo”**. El plato sale igual de bueno cada vez, lo prepare quien lo prepare.

### 2.3 Qué ganamos al llevarlo a la nube

| Antes (laptop / archivo) | Después (plataforma en la nube) | Beneficio para el negocio |
|---|---|---|
| El pronóstico vive en una PC | Vive en un **servicio disponible** que el negocio consulta cuando lo necesita | Decisiones a tiempo, sin depender de quién esté en la oficina |
| Nadie sabe si el modelo aún acierta | Se **mide su precisión automáticamente** (métrica de error vigilada) | Se detecta a tiempo cuando un modelo se “desactualiza” |
| Difícil reconstruir un resultado | Cada modelo queda **registrado y versionado** (qué datos, cuándo, qué tan bueno) | Trazabilidad y respaldo ante auditoría o dudas |
| Depende de una persona | El proceso está **automatizado y documentado** | Continuidad del negocio; no hay “punto único de falla” |
| Una laptop, un producto a la vez | Capacidad de **escalar** a más productos/tiendas | Crece sin rehacer todo desde cero |
| Sin control de costo | **Se enciende y apaga solo** por horario (ver cuadros) | Se paga solo por el uso real (~$70/mes hoy) |

### 2.4 Un caso concreto: el lunes en la mañana

Imaginemos que el área comercial necesita el pronóstico de la semana un lunes a las 8:00 a.m.

**En la computadora del analista (forma tradicional):**
1. Hay que esperar a que esa persona llegue y prenda su laptop.
2. Abre sus archivos, corre el proceso “a mano”, espera a que termine.
3. Copia el resultado a un Excel y lo envía por correo.
4. Si su laptop falla, si está de licencia o si cambió algo en sus archivos, **el pronóstico no sale ese día**.
5. Nadie más puede generar ese número en su ausencia.

**En el servidor (la nube):**
1. La plataforma ya generó y guardó el pronóstico de forma automática, en horario.
2. Cualquier persona autorizada lo consulta desde el dashboard, sin depender de quién esté.
3. El sistema **ya verificó** qué tan preciso viene el modelo y avisa si algo se salió de rango.
4. Queda registro de qué modelo se usó y con qué datos, por si hay que sustentarlo después.

La diferencia no es “tener una computadora más potente”, sino **dejar de depender de una persona y una máquina** para que el negocio reciba siempre el dato, a tiempo y con respaldo.

### 2.5 ¿La computadora tradicional no servía para nada?

Sí servía —y sigue sirviendo— para **explorar, probar ideas y entrenar** un primer modelo. El problema no es entrenar; es **operar** ese modelo todos los días de forma confiable. La nube no reemplaza el trabajo del analista: lo **multiplica y lo protege**, liberándolo de tareas manuales y repetitivas para que se concentre en mejorar los modelos en lugar de “correrlos a mano”.

### 2.6 Por qué entonces existe un costo mensual

Tener un “local abierto y listo para atender” tiene un costo base, igual que mantener una tienda con luz y vigilancia aunque no haya clientes en ese minuto. En nuestro caso ese piso es de **~$58/mes** (la puerta de entrada segura y el repartidor de tráfico que están siempre activos). El resto del costo **sube o baja según cuántas horas mantenemos la plataforma encendida**, y por eso el apagado automático es tan importante: convierte un costo fijo en uno controlable.

> **En una frase:** no estamos pagando por “tener servidores”, estamos pagando por **convertir un pronóstico frágil y dependiente de una persona en un servicio confiable, medible y siempre disponible** — por menos de lo que cuesta un par de horas de consultoría al mes.

---

## 3. Cuadro de costos detallado por escenario

Los costos se separan en **fijos** (corren 24/7, no dependen del horario) y **variables** (se apagan con el scheduler y escalan según las horas de uso).

| Componente | Tipo | A · 24/7 | B · 6d×8h | C · 3d×4h |
|---|---|---:|---:|---:|
| NAT Gateway (+ egress de datos) | Fijo | $33.30 | $33.30 | $33.30 |
| Balanceador ALB (+ LCU) | Fijo | $16.93 | $16.93 | $16.93 |
| RDS — almacenamiento 20 GB gp3 | Fijo | $2.30 | $2.30 | $2.30 |
| CloudWatch (logs + métricas MAPE) | Fijo | $3.21 | $3.21 | $3.21 |
| Transferencia de datos (salida ALB) | Fijo | $0.45 | $0.45 | $0.45 |
| S3 (datos + artefactos + modelos) | Fijo | $0.35 | $0.35 | $0.35 |
| ECR (5 repos de imágenes) | Fijo | $0.35 | $0.35 | $0.35 |
| Lambdas + EventBridge + SNS | Fijo | $0.21 | $0.21 | $0.21 |
| Entrenamientos Batch (~10/mes, Spot) | Demanda | $1.02 | $1.02 | $1.02 |
| **Subtotal fijo** | | **$58.12** | **$58.12** | **$58.12** |
| RDS — cómputo (`db.t4g.small`) | Variable | $23.36 | $6.66 | $1.66 |
| Fargate — MLflow + Reports + API + UI | Variable | $144.16 | $41.08 | $11.93 |
| **Subtotal variable** | | **$167.52** | **$47.73** | **$11.93** |
| **TOTAL MENSUAL** | | **$225.64** | **$105.85** | **$70.05** |

### 3.1 Visual: de dónde viene el costo (piso fijo vs. variable)

**Composición del costo mensual** — cada cuadro ≈ **$14**:

| Escenario | Composición del costo &nbsp; (🟦 piso fijo · 🟥 variable) | Total |
|---|---|---:|
| **A · 24/7** | 🟦🟦🟦🟦 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥 | **$226** |
| **B · 6d×8h** | 🟦🟦🟦🟦 🟥🟥🟥 | **$106** |
| **C · 3d×4h** &nbsp;*(actual)* | 🟦🟦🟦🟦 🟥 | **$70** |

> 🟦 **Piso fijo (~$58)** → NAT Gateway + ALB, siempre activos; **no baja** al apagar.
> 🟥 **Variable** → RDS + Fargate; **se apagan solos** fuera de horario.
>
> **Lectura:** el piso azul (~$58) es idéntico en los tres escenarios; **todo el ahorro** viene de recortar la parte roja, manteniendo la plataforma encendida solo durante las horas de uso real.

---

## 4. Sustento de los costos (cómo se calcula)

**Supuestos de tiempo:**
- Mes estándar AWS = **730 horas** (24 h × 365 ÷ 12).
- Conversión semana→mes = **× 4.33** (52 semanas ÷ 12 meses).
  - Escenario A: 730 h/mes · Escenario B: 6 × 8 × 4.33 = **208 h** · Escenario C: 3 × 4 × 4.33 = **52 h**.

**Tarifas unitarias aplicadas (AWS `us-east-1`, On-Demand):**

| Recurso | Tarifa | Cálculo del componente variable |
|---|---|---|
| Fargate vCPU | $0.04048 / vCPU-hora | — |
| Fargate memoria | $0.004445 / GB-hora | — |
| **Fargate MLflow** (2 vCPU / 4 GB) | $0.09874 / h | × horas/mes |
| **Fargate Reports** (0.5 vCPU / 1 GB) | $0.024685 / h | × horas/mes |
| **Fargate API** (1 vCPU / 2 GB) | $0.04937 / h | × horas/mes |
| **Fargate UI** (0.5 vCPU / 1 GB) | $0.024685 / h | × horas/mes |
| **RDS `db.t4g.small`** (cómputo) | $0.032 / h | × horas/mes |
| ALB | $0.0225 / h × 730 h | = $16.43 (24/7 fijo) |
| NAT Gateway | $0.045 / h × 730 h | = $32.85 (24/7 fijo) |
| RDS almacenamiento gp3 | $0.115 / GB-mes × 20 GB | = $2.30 (siempre) |

**Costo por hora de encendido** (lo que se ahorra al apagar): **$0.2295/h** = $0.19748 (4 servicios Fargate) + $0.032 (RDS). Cada hora apagada ahorra ~$0.23.

**Notas:**
- El RDS (`db.t4g.small`, 2 GB) hospeda **dos bases**: el registro de modelos de MLflow y la base de pronósticos (`forecasts`) de la API. Por eso es `small` y no `micro`.
- Las **5 imágenes** de contenedor (entrenador, MLflow, reports, API, UI) viven en ECR; su costo de almacenamiento es marginal (~$0.35/mes).
- Los entrenamientos corren en **EC2 Spot** (`c6i.2xlarge`, ~$0.10/h) y solo se pagan mientras entrenan; se asumen ~10 entrenamientos/mes de ~1 h. Este costo es por demanda, no por horario.
- Métricas CloudWatch: 6 variedades + 3 métricas base = 9 series × $0.30.

---

## 5. Conclusiones para la decisión

1. **Existe un piso de ~$58/mes** que no baja por apagar servicios: lo imponen el **NAT Gateway (~$33)** y el **ALB (~$17)**, que operan 24/7. Es el costo de mantener la plataforma “lista para encender”.
2. **El apagado automático es el principal ahorro:** pasar de 24/7 a la operación actual (3 días × 4 h) reduce el gasto de **~$226 a ~$70/mes (–69 %)**, es decir **~$1,872/año de ahorro**.
3. **Escalar el horario es lineal y predecible:** cada hora adicional encendida cuesta **~$0.23**. Duplicar el uso (Escenario B, 208 h) sube el total a ~$106/mes, todavía **53 % por debajo** de 24/7.
4. **Para reducir aún más** (fuera del alcance operativo actual): reemplazar el NAT Gateway por *VPC endpoints* recortaría hasta ~$32/mes del piso fijo.

> *Cifras estimadas con tarifas públicas de AWS `us-east-1` a la fecha del reporte; el gasto real puede variar ±15 % según volumen de entrenamientos, tráfico y transferencia de datos. Fuente del modelo: sección 9 (Costos detallados) de la guía MLOps del proyecto.*
