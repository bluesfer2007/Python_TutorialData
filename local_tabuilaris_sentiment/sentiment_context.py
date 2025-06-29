from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

#configuración del modelo offline
ruta_modelo = "./modelo_descargado"  # Ruta donde se guardó el modelo
print(f"Cargando el modelo y el tokenizador desde '{ruta_modelo}'...")

# Cargar el tokenizador y el modelo desde la ruta local
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)

#cargar el modelo desde la ruta local
model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
print("Modelo y tokenizador cargados correctamente.")

# Función para predecir el sentimiento de un texto
def predecir_sentimiento(texto):
    # Tokenizar el texto
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener las probabilidades de cada clase
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicciones=torch.argmax(probabilities, dim=-1).tolist()

    #mapear los sentimientos 
    mapa_sentimientos = {
        0: "Muy Negativo",
        1: "Negativo",
        2: "Neutro",    
        3: "Positivo",
        4: "Muy Positivo"
    }
    
    return [mapa_sentimientos[pred] for pred in predicciones]

#leer los datos del csv
df= pd.read_csv("../test_comentarios.csv",sep=";")
df['sentimiento'] = df['Texto'].apply(predecir_sentimiento)

print(df.head())