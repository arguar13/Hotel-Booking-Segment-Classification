*Leer este documento en otros idiomas: [English](README.md)*

# Demanda de Reservas Hoteleras: Clasificación Multiclase de Segmentos de Mercado

## 1. Descripción General del Proyecto

Este repositorio contiene un pipeline de Machine Learning de extremo a extremo que predice el **market_segment** de una reserva hotelera **en el momento en que se realiza**. Usando solo información disponible al reservar (detalles de la estadía, composición del grupo, anticipación, tarifa, tipo de habitación y de depósito, historial del huésped), el modelo clasifica cada reserva en uno de siete segmentos de mercado.

Una clasificación precisa de los segmentos permite auditar el etiquetado, completar segmentos faltantes, personalizar precios y marketing, y pronosticar la demanda por segmento.

---

## 2. Contexto de Negocio

En la industria hotelera, comprender el origen de cada reserva es fundamental. Los segmentos difieren en comportamiento de cancelación, anticipación, duración de la estadía y sensibilidad al precio, por lo que las decisiones de revenue management dependen de saber a qué segmento pertenece cada reserva.

---

## 3. Información del Conjunto de Datos

El proyecto utiliza el conjunto de datos público **Hotel Booking Demand** (Antonio, Almeida y Nunes, 2019), con reservas de un hotel urbano en Lisboa y un resort en el Algarve, con llegadas entre 2015 y 2017.

### Variable Objetivo

**market_segment** (7 clases, fuertemente desbalanceadas):

| Segmento | Proporción |
|----------|-----------:|
| Online TA | 59,1% |
| Offline TA/TO | 15,9% |
| Direct | 13,5% |
| Groups | 5,7% |
| Corporate | 4,8% |
| Complementary | 0,8% |
| Aviation | 0,26% |

### Alcance de las Variables

El modelo está diseñado para **predecir al momento de la reserva**, por lo que se excluyen deliberadamente dos grupos de columnas:

| Columnas excluidas | Motivo |
|--------------------|--------|
| `is_canceled`, `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes` | Solo se conocen después de realizada la reserva (información futura) |
| `distribution_channel`, `agent`, `company` | Identificadores de canal que codifican el segmento casi uno a uno |

---

## 4. Metodología y Arquitectura

El flujo de trabajo prioriza la prevención de fuga de datos (data leakage), una evaluación honesta y la reproducibilidad.

### 4.1 Limpieza de Datos

- Eliminación de 31.994 registros duplicados antes de cualquier división.
- Eliminación de clases con menos de 10 registros (`Undefined`) para permitir la división estratificada.
- Conjunto final: 87.394 reservas.

### 4.2 Estrategia de Evaluación

División estratificada en **Train (64%)**, **Validation (16%)** y **Test (20%)**. Los modelos se comparan y ajustan solo sobre Validation; Test se usa una única vez, para la evaluación final.

### 4.3 Preprocesamiento e Ingeniería de Características

Transformadores personalizados de Scikit-Learn integrados en un único pipeline:

- **FeatureEngineer:** noches totales, huéspedes totales e indicador de niños; codificación cíclica (seno/coseno) del mes y de la semana de llegada.
- **IQRTransformer:** recorta valores atípicos a `[Q1 − 1,5·IQR, Q3 + 1,5·IQR]`, aprendidos solo con datos de entrenamiento. Las columnas con IQR nulo (indicadores binarios, conteos dispersos) no se modifican.
- **RareCategoryEncoder:** agrupa en `"Other"` las categorías con frecuencia menor al 1% antes del One-Hot Encoding.

Las columnas numéricas y categóricas se detectan automáticamente con `make_column_selector`.

### 4.4 Desbalance de Clases

**Sobremuestreo moderado con SMOTE** dentro de un `Pipeline` de imbalanced-learn: cada clase minoritaria se eleva al menos al 15% de la clase mayoritaria. El remuestreo ocurre solo durante `fit()`, por lo que validación y test conservan siempre la distribución real de clases.

### 4.5 Comparación de Modelos (Benchmarking)

Cinco algoritmos se entrenan con el mismo pipeline y se ordenan en validación por **F1 macro** (todos los segmentos pesan lo mismo); el F1 ponderado se reporta junto a él:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### 4.6 Optimización de Hiperparámetros

El mejor modelo se selecciona automáticamente y se ajusta con **Optuna** (muestreador TPE, 25 pruebas, F1 macro en validación). Los parámetros ajustados solo se adoptan si superan a la configuración del benchmark. El modelo final se reentrena con Train + Validation antes de la evaluación en test.

---

## 5. Resultados

### Benchmark en validación

| Modelo | F1 macro | F1 ponderado |
|--------|---------:|-------------:|
| **Random Forest** | **0,754** | **0,805** |
| XGBoost | 0,747 | 0,803 |
| LightGBM | 0,744 | 0,795 |
| Gradient Boosting | 0,648 | 0,735 |
| Logistic Regression | 0,505 | 0,661 |

Optuna elevó el F1 macro de Random Forest en validación de 0,754 a 0,759.

### Conjunto de test final (17.479 reservas no vistas)

| Segmento | Precision | Recall | F1 |
|----------|----------:|-------:|---:|
| Online TA | 0,84 | 0,94 | 0,89 |
| Groups | 0,87 | 0,91 | 0,89 |
| Complementary | 0,80 | 0,90 | 0,85 |
| Offline TA/TO | 0,85 | 0,72 | 0,78 |
| Corporate | 0,75 | 0,76 | 0,75 |
| Aviation | 0,68 | 0,71 | 0,70 |
| Direct | 0,68 | 0,43 | 0,53 |
| **Promedio macro** | **0,78** | **0,77** | **0,77** |
| **Promedio ponderado** | **0,82** | **0,83** | **0,82** |

Accuracy: 0,83. El ROC-AUC one-vs-rest va de 0,881 (Direct) a 0,999 (Complementary).

**Hallazgos clave**

- Los segmentos con un perfil de reserva distintivo (Groups, Complementary, Aviation) se reconocen bien incluso con pocos ejemplos.
- El principal desafío es **Direct frente a Online TA**: casi la mitad de las reservas directas se predicen como reservas de agencia online. Sin información del canal, ambas se parecen al mismo huésped de ocio, lo que marca el límite natural de la predicción al momento de reservar.

---

## 6. Estructura del Proyecto

```
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   └── Hotel Booking Demand.ipynb
├── requirements.txt
├── README.md
└── README_es.md
```

---

## 7. Stack Tecnológico

| Categoría | Tecnologías |
|-----------|-------------|
| Lenguaje | Python 3.12 |
| Manipulación y Análisis de Datos | Pandas, NumPy |
| Framework de Machine Learning | Scikit-Learn |
| Datos Desbalanceados | imbalanced-learn (SMOTE) |
| Ensambles Avanzados | XGBoost, LightGBM |
| Optimización de Hiperparámetros | Optuna |
| Visualización de Datos | Matplotlib, Seaborn |

---

## 8. Guía de Ejecución

1. Clone el repositorio e instale las dependencias:

```bash
git clone https://github.com/arguar13/Hotel-Booking-Segment-Classification.git
cd Hotel-Booking-Segment-Classification
pip install -r requirements.txt
```

2. Abra `notebooks/Hotel Booking Demand.ipynb` y ejecute todas las celdas. El notebook encuentra `data/hotel_bookings.csv` tanto si se lanza desde la raíz del proyecto como desde `notebooks/`.

Una ejecución completa (benchmark + 25 pruebas de Optuna) tarda unos 30 minutos en una laptop estándar.

---

## Licencia

Proyecto educativo y de portafolio de Ciencia de Datos. Conjunto de datos obtenido de Kaggle y utilizado con fines no comerciales.

---

## Autor

**Armando Guarnera**  
Científico de Datos  
Argentina
