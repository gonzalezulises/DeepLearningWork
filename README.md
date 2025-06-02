# Proyecto Final: Predicción del Éxito de Atracciones Turísticas con Deep Learning Híbrido

## 1. Introducción y Objetivo del Proyecto

El presente proyecto, desarrollado como parte del Módulo de Deep Learning de KeepCoding, tiene como **objetivo principal** desarrollar e implementar un modelo avanzado de Deep Learning capaz de predecir con precisión el nivel de engagement (alto o bajo) que generarán distintos Puntos de Interés (POIs) turísticos[cite: 81].

Para lograr esto, el modelo integra de manera innovadora dos fuentes de información complementarias[cite: 82]:
* **Características visuales**: Extraídas directamente de las imágenes representativas de cada POI.
* **Metadatos estructurados**: Información contextual asociada a cada POI.

Los datos utilizados provienen de la plataforma **Artgonuts**[cite: 83], y las imágenes han sido específicamente procesadas para esta práctica, con sus versiones originales procedentes de fuentes como el portal de datos abiertos de la Comunidad de Madrid[cite: 84]. El desafío central es anticipar la interacción que cada POI generará, lo cual representa un activo estratégico para plataformas turísticas al permitir optimizar la selección de contenido y mejorar la experiencia del usuario[cite: 85, 86].

El modelo final actúa como un clasificador binario, determinando si un POI generará un engagement "alto" o "bajo"[cite: 88].

## 2. Entorno de Desarrollo y Reproducibilidad

Este proyecto ha sido desarrollado íntegramente en **Google Colab**, aprovechando sus recursos de GPU.

### 2.1. Configuración para la Reproducción
Para reproducir este proyecto, sigue estos pasos:

1.  **Cuenta de Google:** Necesitarás una cuenta de Google para usar Colab y Google Drive.
2.  **Google Drive:**
    * Crea la siguiente estructura de carpetas en tu Google Drive (por ejemplo, dentro de `My Drive/Colab Notebooks/`):
        ```
        Proyecto_POI_Engagement/
        ├── data/
        ├── trained_models/
        └── memoria_tecnica
        ├── notebooks/
        ```
    * Descarga el dataset proporcionado para la práctica. Este debería incluir un archivo `poi_dataset.csv` y un archivo `data_main.zip` (que contiene las imágenes en una estructura de carpetas por ID).
    * Sube `poi_dataset.csv` y `data_main.zip` a la carpeta `Proyecto_POI_Engagement/data/` en tu Drive.
3.  **Google Colab:**
    * Abre el archivo `.ipynb` de este proyecto en Google Colab.
    * Habilita la **GPU**: Ve a `Entorno de ejecución` -> `Cambiar tipo de entorno de ejecución` y selecciona `GPU` como acelerador por hardware.
4.  **Dependencias:**
    * Las principales bibliotecas utilizadas (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`, `torch`, `torchvision`) suelen estar preinstaladas en Colab.
    * Si alguna biblioteca adicional o versión específica fue necesaria (como `captum` para Grad-CAM si se implementó), el notebook incluirá una celda al principio con los comandos `!pip install ...` correspondientes. (Se recomienda consultar el archivo `requirements.txt` si se proporciona junto al proyecto para un entorno local [cite: 54, 55]).
5.  **Ejecución:**
    * La primera celda de código ejecutable montará tu Google Drive.
    * Asegúrate de que la variable `DRIVE_PROJECT_PATH` en el notebook apunte correctamente a la carpeta `Proyecto_POI_Engagement` en tu Drive.
    * Ejecuta las celdas del notebook en orden secuencial. Se utilizan semillas de aleatoriedad (`SEED = 42`) para garantizar la reproducibilidad de los resultados en la medida de lo posible[cite: 56].

## 3. Metodología y Desarrollo del Proyecto

Siguiendo la estructura propuesta en el enunciado, el proyecto se dividió en las siguientes fases:

### 3.1. Preparación y Análisis de Datos (25%) [cite: 109]

Esta fase fue crucial para entender y preparar los datos para el modelado.

* **Carga Inicial:**
    * Se cargó `poi_dataset.csv`, resultando en un DataFrame con 1569 POIs y 14 características.
    * *Conclusión:* Los datos se cargaron correctamente. Una inspección inicial con `.info()` indicó la ausencia de valores nulos directos en las columnas[cite: 111].

* **Limpieza de Datos - IDs Duplicados:**
    * Se identificaron 23 `id` que aparecían múltiples veces (el más frecuente, 7 veces). Estas filas diferían principalmente en la columna `tags` y ligeramente en las métricas de engagement.
    * Se optó por consolidar estas entradas por `id`, promediando las métricas numéricas de engagement y creando una lista única de `tags` para cada `id`.
    * *Conclusión:* El DataFrame se redujo a 1492 POIs únicos, proporcionando una base de datos más limpia y coherente para el modelado.

* **Análisis Exploratorio de Datos (EDA) - Metadatos (Post-Consolidación):**
    * `categories` y `tags`: Se parsearon de su formato string original (ej., `"['Arte', 'Historia']"`) a listas de Python (`categories_parsed` y `tags_parsed`).
        * `categories_parsed` mostró una baja cardinalidad (12 categorías únicas como 'Patrimonio', 'Cultura'), lo que sugiere que técnicas como one-hot encoding o embeddings simples serían viables[cite: 112, 116].
        * `tags_parsed` reveló una alta cardinalidad (2935 tags únicos), indicando la necesidad de embeddings para su manejo, como se advierte en el enunciado contra el one-hot encoding de miles de categorías[cite: 117].
    * `Visits`: Esta columna presentó un rango de valores extremadamente estrecho (ej., 10001-10038) incluso después de la consolidación, lo que la hizo poco informativa para diferenciar el engagement y se decidió excluirla de la creación del score de engagement.
    * `locationLon`, `locationLat`: Mostraron una dispersión global, pero con una concentración significativa (~70%) en el área aproximada de Madrid. Las columnas `distrito` y `barrio`, aunque mencionadas en la descripción del dataset del PDF[cite: 97, 171], no estaban presentes en el archivo CSV utilizado.
    * Otras métricas como `Likes` y `Bookmarks` mostraron buena varianza, siendo candidatas para la métrica de engagement.
    * *Conclusión:* El EDA fue fundamental para comprender la estructura, calidad y particularidades de los datos, guiando las decisiones de preprocesamiento y selección de características.

* **EDA - Imágenes:**
    * Se verificó que el 100% de los POIs tenían una ruta de imagen (`main_image_path`) válida y accesible, con un formato `data_main/POI_ID/main.jpg`.
    * Un análisis de una muestra de 200 imágenes confirmó que todas ya estaban preprocesadas a un tamaño uniforme de **128x128 píxeles** y en modo **RGB**.
    * *Conclusión:* Las imágenes estaban "específicamente procesadas para los fines de esta práctica"[cite: 84], lo que simplificó enormemente el preprocesamiento visual, no requiriendo redimensionamiento ni conversión de modo de color generalizados[cite: 110].

* **Creación de Métrica de Engagement (`target_engagement`):** [cite: 113]
    * Se desarrolló una variable objetivo binaria. Se utilizaron las métricas `Likes` y `Bookmarks`. Estas fueron transformadas (log1p), escaladas (MinMaxScaler) y luego promediadas para crear un `engagement_score`.
    * Este score se binarizó utilizando su mediana como umbral, resultando en la variable `target_engagement` (0 = Bajo, 1 = Alto).
    * *Conclusión:* Se creó exitosamente una variable objetivo binaria con clases perfectamente balanceadas (50%/50%), lo cual es óptimo para el entrenamiento de un modelo de clasificación[cite: 119].

* **División Estratificada del Dataset:** [cite: 114]
    * Los datos se dividieron en conjuntos de entrenamiento (60%), validación (20%) y prueba (20%), manteniendo la proporción 50/50 de la variable objetivo en cada conjunto.
    * *Conclusión:* Se aseguraron particiones de datos adecuadas para un entrenamiento y evaluación robustos del modelo.

### 3.2. Arquitectura del Modelo (35%) [cite: 120]

Se diseñó e implementó un modelo híbrido en PyTorch.

* **PyTorch `Dataset` y `DataLoader`:**
    * Se creó una clase `PoiDataset` personalizada para cargar las imágenes (aplicando transformaciones como `ToTensor` y `Normalize` con medias/std de ImageNet) y los metadatos correspondientes (numéricos, secuencias de categorías y tags).
    * Se generaron `DataLoader`s para los conjuntos de entrenamiento, validación y prueba con un `BATCH_SIZE = 32`.
    * *Conclusión:* Se estableció un pipeline de datos eficiente y correcto, con tensores de formas y tipos de datos esperados para alimentar el modelo.

* **Definición del Modelo Híbrido (`HybridPoiModel`):** [cite: 121]
    * **Componente Visual:** Se utilizó `ResNet18` pre-entrenado, con sus capas convolucionales iniciales congeladas, actuando como extractor de características. Su salida se procesó mediante una capa lineal para obtener un vector de 256 características visuales.
    * **Componente Contextual:**
        * *Numéricos:* (6 características: `tier`, `locationLon`, `locationLat`, `xps`, `num_categories_parsed`, `num_tags_parsed`) procesados por una MLP.
        * *`categories_parsed`:* (Vocabulario ~14) procesados mediante una capa `nn.Embedding` (dimensión 10), seguida de un promedio de los embeddings y una MLP.
        * *`tags_parsed`:* (Vocabulario ~1000) procesados mediante una capa `nn.Embedding` (dimensión 50), seguida de un promedio de los embeddings y una MLP.
        * Las salidas de estas sub-ramas se concatenaron y pasaron por una MLP final, produciendo un vector de 128 características de metadatos.
    * **Fusión:** Las características visuales (256) y de metadatos (128) se concatenaron, resultando en un vector de 384 características.
    * **Cabezal de Clasificación:** Una MLP con capas de Dropout tomó el vector fusionado y produjo un único logit para la clasificación binaria.
    * *Conclusión:* Se implementó una arquitectura híbrida funcional que integra ambas modalidades de datos, siguiendo las directrices del enunciado. Las dimensiones de las capas y las conexiones fueron verificadas.

### 3.3. Entrenamiento y Optimización (25%) [cite: 203]

* **Configuración:** Se utilizó `nn.BCEWithLogitsLoss` como función de pérdida y el optimizador `Adam` con una tasa de aprendizaje de `0.001`. El entrenamiento se realizó en GPU (`cuda`).
* **Proceso de Entrenamiento:** Se entrenó el modelo durante 10 épocas. Se implementó el guardado del mejor modelo (`model checkpointing`) basado en la menor pérdida de validación.
* **Resultados del Entrenamiento Inicial:**
    * El modelo demostró un aprendizaje efectivo, con una disminución de la pérdida y un aumento de la precisión tanto en entrenamiento como en validación a lo largo de las épocas.
    * La mejor pérdida de validación fue de `0.2497` y la precisión de validación correspondiente fue de `91.64%` (alcanzada en la época 10).
    * No se observaron signos de overfitting severo en estas 10 épocas. Se manejó un error de carga para una imagen corrupta durante el entrenamiento haciendo que el `Dataset` devolviera tensores de ceros, permitiendo que el proceso continuara.
    * *Conclusión:* El modelo se entrenó exitosamente y su rendimiento superó claramente al de un clasificador aleatorio, cumpliendo los objetivos de esta fase[cite: 203]. Las técnicas anti-overfitting básicas (Dropout, monitorización de validación) estuvieron presentes[cite: 126].

### 3.4. Evaluación y Análisis (15%) [cite: 207]

* **Carga del Mejor Modelo:** El modelo con la mejor pérdida de validación fue cargado para la evaluación en el conjunto de prueba.
* **Rendimiento en el Conjunto de Prueba:**
    * Pérdida en Test: **`0.1770`**
    * Precisión en Test: **`0.9331` (93.31%)** (o `93.65%` según `classification_report`)
    * AUC: **`0.9755`**
    * El reporte de clasificación detallado mostró:
        * Para "Bajo Engagement (0)": Precision=0.9119, Recall=0.9667, F1-score=0.9385
        * Para "Alto Engagement (1)": Precision=0.9643, Recall=0.9060, F1-score=0.9343
    * *Conclusión Principal:* El modelo generalizó excelentemente a datos no vistos, con un rendimiento en el conjunto de prueba incluso ligeramente superior al de validación. Esto indica un modelo robusto y fiable[cite: 207].

* **Análisis de Errores y Casos Específicos:** [cite: 210]
    * La matriz de confusión (`[[145 TN, 5 FP], [14 FN, 135 TP]]`) mostró un bajo número de Falsos Positivos (5) y un número ligeramente mayor pero aún bajo de Falsos Negativos (14).
    * Se inspeccionaron cualitativamente algunos ejemplos de estos errores (imágenes y metadatos).
    * *Conclusión:* Este análisis ayuda a comprender las limitaciones del modelo y los tipos de POIs donde podría fallar, ofreciendo pistas para futuras mejoras.

* **Interpretabilidad del Modelo y Visualización de Features Importantes:** [cite: 209]
    * **Metadatos (Permutation Importance):**
        * `category_features` demostraron ser el grupo de metadatos más influyente (caída de AUC de 0.2346 al permutarlas).
        * `numerical_features` tuvieron un impacto moderado (caída de AUC de 0.0032).
        * `tag_features` mostraron el menor impacto (caída de AUC de 0.0020) en la configuración actual.
        * *Conclusión:* La categorización principal del POI es un predictor de metadatos muy fuerte para el engagement.
    

* **Propuestas de Mejoras Futuras:** [cite: 213]
    * Fine-tuning más exhaustivo del componente CNN (descongelando más capas).
    * Optimización de hiperparámetros avanzada (tasas de aprendizaje, arquitecturas de MLP, dimensiones de embedding, dropout).
    * Incorporación de características textuales de `name` y `shortDescription` mediante técnicas NLP más avanzadas (ej., embeddings de Transformers).
    * Investigación adicional sobre la columna `Visits` o su reemplazo.
    * Experimentación con arquitecturas de fusión más complejas (ej., atención).
    * Aplicación de Data Augmentation para imágenes.
    * Profundización en técnicas de interpretabilidad como SHAP.
    * *Conclusión:* Se identificaron múltiples vías para la optimización y mejora continua del modelo.

## 4. Conclusión General del Proyecto

El proyecto ha culminado con el desarrollo exitoso de un modelo híbrido de Deep Learning capaz de predecir el engagement de Puntos de Interés Turístico con una alta precisión (93.31% - 93.65% en el conjunto de prueba) y un excelente AUC (0.9755). El modelo integra eficazmente información visual y de metadatos, cumpliendo con los objetivos principales establecidos en el enunciado[cite: 81, 82, 85].

El proceso ha abarcado un análisis exhaustivo de los datos, un preprocesamiento cuidadoso, el diseño de una arquitectura de red neuronal apropiada, un entrenamiento efectivo y una evaluación detallada. Los resultados demuestran no solo la viabilidad del enfoque, sino también su potencial como herramienta estratégica para plataformas del sector turístico[cite: 86].

Más allá de las métricas, esta práctica ha permitido desarrollar una comprensión profunda de las técnicas de deep learning y su aplicación a un problema del mundo real, incluyendo el manejo de datos multimodales, la interpretabilidad y la mejora iterativa[cite: 79]. Las propuestas de mejoras futuras abren el camino para continuar explorando y refinando esta solución.

## 5. Entregables del Proyecto (Referencia al Enunciado)

Este proyecto se estructura para cumplir con los siguientes entregables clave[cite: 54]:
1.  **Notebook con código comentado y reproducible:** El presente notebook `.ipynb`. (Se debe generar un `requirements.txt` a partir del entorno final). [cite: 54, 55]
2.  **Memoria técnica detallada (PDF):** Un documento separado que explica exhaustivamente todo el proceso. [cite: 57, 59]
3.  **Modelo entrenado final:** El archivo `best_hybrid_poi_model.pth` guardado en la carpeta `trained_models/`. [cite: 60]
4.  **Scripts de preprocesamiento y utilidades:** En este caso, la mayoría del código está contenido dentro de este notebook, pero cualquier función auxiliar reutilizable podría extraerse. [cite: 61]

---
