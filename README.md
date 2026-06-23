# Del Excel al Modelo: pricing con un GLM

## PwC Acceleration Center Buenos Aires

Por [Bruno Celia](https://www.linkedin.com/in/bruno-celia/) y
[Nicolás Leonardo Chavez](https://www.linkedin.com/in/nicolas-leonardo-chavez)

Material de la charla **"Herramientas de análisis de datos para actuarios"** — CPCE.

En esta demo tomamos un CSV crudo de costos médicos, lo entendemos, y construimos un
modelo de **pricing explicable, auditable y defendible** usando un **GLM (Generalized
Linear Model)** — con poco SQL y poco Python.

> **La pregunta que responde el modelo:** tenemos a Ana (25, no fuma, peso saludable) y a
> Carlos (55, fuma, sobrepeso, 2 hijos). ¿Les cobrarías lo mismo? ¿Cuánto a cada uno?

---

## 📦 Qué hay en este repo

| Archivo | Qué es |
|---|---|
| `demo_glm_pricing.ipynb` | El notebook de la demo, completo y comentado |
| `insurance.csv` | El dataset (1.338 asegurados, 7 columnas) |
| `README.md` | Esta guía |

---

## 🎯 Qué vas a aprender

- Por qué un **GLM Gamma con link logarítmico** es la herramienta correcta para pricing
  (y no una regresión lineal común).
- Cómo leer las **relatividades** (los factores multiplicativos) y defenderlas ante un regulador.
- Por qué **significancia estadística ≠ poder predictivo** (la trampa más común).
- Cómo validar un modelo de forma honesta (train/test) y reconocer **overfitting** y **dato sucio**.

---

## 🚀 Cómo correrlo (gratis, sin instalar nada)

Recomendamos **Databricks Free Edition** (antes "Community Edition"): es gratis, no pide
tarjeta, y te da todo lo necesario para correr la demo.

### 1. Creá tu cuenta gratis
Entrá a 👉 **https://www.databricks.com/learn/free-edition** y registrate con tu email.
En un par de minutos tenés un workspace listo.

### 2. Descargá el material de este repo
- Clic en el botón verde **`Code` → Download ZIP** (arriba de esta página), o
- Cloná el repo: `git clone https://github.com/rms-modernization/charla-cpce.git`

Necesitás dos archivos: `demo_glm_pricing.ipynb` y `insurance.csv`.

### 3. Importá el notebook a Databricks
En tu workspace: **Workspace → (tu carpeta) → Import** → subí `demo_glm_pricing.ipynb`.

### 4. Subí el dataset
La forma más simple en Free Edition:
1. **Catalog → (tu catálogo) → (tu schema) → Create → Table → Upload file**
2. Subí `insurance.csv` → se crea una tabla, por ejemplo `tu_catalog.tu_schema.insurance`.

### 5. Ajustá la ruta del dato en el notebook
La primera celda de código carga el dato. Editala según cómo subiste el CSV:

```python
# Si lo subiste como TABLA de Unity Catalog (recomendado):
sdf = spark.table("tu_catalog.tu_schema.insurance")
df  = sdf.toPandas()

# O, si preferís leerlo como archivo suelto:
# df = pd.read_csv("/Volumes/tu_catalog/tu_schema/tu_volume/insurance.csv")

# O, para correrlo LOCAL en tu compu (fuera de Databricks):
# df = pd.read_csv("insurance.csv")
```

> Reemplazá `tu_catalog`, `tu_schema` y `tu_volume` por los nombres reales de tu workspace.

### 6. Ejecutá todo
**Run all** y listo. Los gráficos, las tablas y las primas de Ana y Carlos aparecen solos.

---

## 🛠️ ¿Y si me faltan librerías?

El runtime de Databricks ya trae casi todo. Si algo falla (por ejemplo MLflow en
*serverless*), agregá esta celda al **principio** del notebook y ejecutala primero:

```python
%pip install -q "mlflow>=2.20,<3" "statsmodels>=0.14" scikit-learn
dbutils.library.restartPython()
```

Después de eso, ejecutá el resto del notebook desde el inicio.

---

## 🏋️ Bonus: tu turno (tarea para el hogar)

La mejor forma de que esto te quede es **ensuciarte las manos**. Tres desafíos, de menor a
mayor dificultad — elegí el que te quede cómodo.

### 🟢 Nivel 1 — Jugá con el modelo que ya tenés
- Cambiá a Ana y Carlos por **vos y un conocido**. Poné tus datos reales y mirá qué prima
  saldría. ¿Te parece justa?
- **Sacá una variable** (por ejemplo `region`) y reentrená. ¿Cambió mucho el R² en test?
- Probá **otra interacción** además de `smoker:bmi` (por ejemplo `age:smoker`). ¿Es
  significativa? ¿Mejora la predicción, o cae en la misma trampa que vimos en clase?

### 🟡 Nivel 2 — Cambiá el motor estadístico
- Probá **otra familia/link**: `InverseGaussian(link=Log())` o `Gamma(link=Identity())`.
  ¿Mejoran las métricas? ¿Alguna vuelve a predecir primas negativas?
- Compará el GLM Gamma contra un **OLS sobre `log(charges)`** (la alternativa log-normal).
  ¿Cuál predice mejor en dólares?

### 🔴 Nivel 3 — Otro problema, otra distribución: frecuencia con Poisson
El salto que hace todo actuario de P&C (auto, hogar):
- Bajá el dataset **French Motor Third-Party Liability** (`freMTPL2freq`), disponible en
  [Kaggle](https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq)
  o en el paquete `CASdatasets` de R.
- En vez de modelar un **monto** (severidad, con Gamma), modelá la **frecuencia** de
  siniestros: un conteo. La distribución correcta para conteos es **Poisson**.
- Pistas para arrancar:
  - Familia: `sm.families.Poisson(link=sm.families.links.Log())`
  - La **exposición** entra como *offset*: `offset=np.log(df["Exposure"])`
  - El target es `ClaimNb` (entero ≥ 0), no un monto.

> **La idea de fondo:** el GLM es un mismo marco — *distribución + link + variables* — que
> se adapta al problema. Severidad (montos con cola) → **Gamma**. Frecuencia (conteos) →
> **Poisson**. Cambia la familia, no el enfoque. Si entendiste la demo, ya tenés todo para
> encarar el de Poisson solo.

---

## 📊 Sobre el dataset

`insurance.csv` — **Medical Cost Personal Dataset** (1.338 asegurados):

| Columna | Qué es |
|---|---|
| `age` | Edad del asegurado |
| `sex` | Sexo |
| `bmi` | Índice de masa corporal |
| `children` | Hijos a cargo |
| `smoker` | ¿Fuma? (yes/no) |
| `region` | Región |
| `charges` | **Gasto médico anual ($)** — lo que predecimos |

Fuente original: [Medical Cost Personal Datasets en Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).

> ℹ️ **Aclaración:** `charges` es el **costo médico** que generó la persona (el siniestro),
> no la prima que paga. El modelo predice el **costo esperado** (la prima de riesgo); la
> prima comercial le suma gastos y margen.

---

## 💬 ¿Dudas o querés compartir tu resultado?

Abrí un *issue* en este repo o escribinos. La mejor forma de aprender es romperlo y arreglarlo.
