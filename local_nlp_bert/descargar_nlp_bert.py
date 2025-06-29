from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Identificador oficial del modelo en Hugging Face
nombre_modelo = "nlptown/bert-base-multilingual-uncased-sentiment"

# Ruta a la carpeta local donde se guardará el nuevo modelo
ruta_local = "./modelo_bert_descargado"

# Crear la carpeta si no existe
if not os.path.exists(ruta_local):
    os.makedirs(ruta_local)

print(f"Descargando el modelo '{nombre_modelo}' a la carpeta '{ruta_local}'...")

# Descargar y guardar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
tokenizer.save_pretrained(ruta_local)

# Descargar y guardar el modelo
model = AutoModelForSequenceClassification.from_pretrained(nombre_modelo)
model.save_pretrained(ruta_local)

print("¡Descarga completada! El modelo de calificación por estrellas está listo para usarse offline.")