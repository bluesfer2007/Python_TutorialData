from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Nombre del modelo en Hugging Face
nombre_modelo = "tabularisai/multilingual-sentiment-analysis"

# Ruta a la carpeta local donde se guardará el modelo
ruta_local = "./modelo_descargado"

print(f"Descargando el modelo y el tokenizador a '{ruta_local}'...")

# Descargar y guardar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
tokenizer.save_pretrained(ruta_local)

# Descargar y guardar el modelo
model = AutoModelForSequenceClassification.from_pretrained(nombre_modelo)
model.save_pretrained(ruta_local)

print("¡Descarga completada! El modelo está listo para usarse offline.")