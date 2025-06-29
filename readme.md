# 🤖 Análisis de Sentimiento Dual con Modelos Locales 📊

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Librerías-Transformers%20%7C%20PyTorch%20%7C%20Pandas-orange.svg)
![License](https://img.shields.io/badge/Licencia-MIT-green.svg)

¡Bienvenido a este proyecto de análisis de sentimiento! 🚀 Este pipeline está diseñado para leer un archivo de comentarios, procesarlos y enriquecerlos usando dos potentes modelos de lenguaje de Hugging Face. Todo el proceso se ejecuta de manera **100% local y offline**, garantizando la máxima velocidad y la total privacidad de tus datos.

## ✨ Características Principales

* **🌐 Ejecución Offline:** Tras una descarga inicial, no se necesita conexión a internet. ¡Analiza tus datos en cualquier lugar!
* **🧠 Análisis Dual:** Cada comentario es evaluado por dos modelos diferentes para obtener una visión más completa:
    * **Clasificación Categórica:** `Muy Positivo`, `Positivo`, `Neutral`, `Negativo`, `Muy Negativo`.
    * **Calificación por Estrellas:** De ★☆☆☆☆ (1 estrella) a ★★★★★ (5 estrellas).
* **⚙️ Procesamiento Eficiente:** Utiliza un sistema de lotes para procesar miles de comentarios sin agotar la memoria de tu equipo.
* **⚡ Aceleración por GPU:** ¡Saca el máximo provecho a tu hardware! Detecta y usa automáticamente tu tarjeta gráfica NVIDIA (CUDA) para un análisis mucho más rápido.
* **📁 Manejo Sencillo de Datos:** Lee archivos `.xlsx` de forma nativa y exporta los resultados a un archivo `.csv` limpio y listo para usar.

## 🤖 Modelos Utilizados

Este proyecto se apoya en dos modelos de lenguaje de código abierto, descargados y ejecutados localmente:

1.  **Tabularis Sentiment Model**
    * **Descripción:** Un modelo multilingüe basado en `DistilBERT` que clasifica el texto en 5 categorías de sentimiento.
    * **Identificador en Hugging Face:** `tabularisai/multilingual-sentiment-analysis`

2.  **NLP Town - Star Rating Model**
    * **Descripción:** Un modelo multilingüe basado en `BERT` entrenado para calificar reseñas en una escala de 1 a 5 estrellas.
    * **Identificador en Hugging Face:** `nlptown/bert-base-multilingual-uncased-sentiment`

## 📂 Estructura del Proyecto

